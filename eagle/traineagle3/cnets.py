# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union
from collections import Counter
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.activations import ACT2FN
from transformers import AutoTokenizer, AutoModelForCausalLM
from configs import EConfig
from safetensors import safe_open
from datasets import load_dataset
import multiprocessing

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)



class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            scaling_type = "default"
            scaling_factor = 1.0
        else:
            if isinstance(self.config.rope_scaling, dict):
                scaling_type = self.config.rope_scaling.get("type", "default")
                scaling_factor = self.config.rope_scaling.get("factor", 1.0)
            else:
                scaling_type = getattr(self.config.rope_scaling, "type", "default")
                scaling_factor = getattr(self.config.rope_scaling, "factor", 1.0)
        if scaling_type == "linear":
            self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
            )
        elif scaling_type == "dynamic":
            self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
            )
        elif scaling_type == "default":
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            cache_hidden: Optional[List[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        lck = len(cache_hidden[0])

        # cache_k = [self.k_proj(hidden) for hidden in cache_hidden]
        # cache_v = [self.v_proj(hidden) for hidden in cache_hidden]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        rotary_seq_len = q_len + lck
        if position_ids is not None:
            rotary_seq_len = max(rotary_seq_len, int((position_ids + lck).max().item()) + 1)
        cos, sin = self.rotary_emb(query_states, seq_len=rotary_seq_len)
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)
        # query_states = apply_rotary_pos_emb(query_states, cos, sin, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids + lck)


        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Avoid modify hidden cache inplace which will cause in-place modification error when enable gradient checkpoint. 
        # Return the updated hidden cache instead.
        if cache_hidden is None:
            local_cache_k = []
            local_cache_v = []
        else:
            local_cache_k = list(cache_hidden[0])
            local_cache_v = list(cache_hidden[1])

        local_cache_k.append(key_states)
        local_cache_v.append(value_states)
            
        cache_k = local_cache_k
        cache_v = local_cache_v

        k0 = cache_k[0]
        v0 = cache_v[0]

        attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(self.head_dim)
        lck = len(cache_k)


        attn_weights = attn_weights + attention_mask

        for i in range(1, lck):
            ki = cache_k[i]

            qi = query_states
            kiq = ki

            attn_weightsi = (qi * kiq).sum(-1) / math.sqrt(self.head_dim)
            attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        first_cache_len = v0.shape[2]
        attn_weights0 = attn_weights[..., :first_cache_len]

        attn_output = torch.matmul(attn_weights0, v0)

        for i in range(1, lck):
            vi = cache_v[i]
            attn_weightsi = attn_weights[..., first_cache_len + i - 1]
            attn_outputi = attn_weightsi[..., None] * vi
            attn_output = attn_output + attn_outputi

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        # Return the updated hidden cache.
        new_past_key_value = [local_cache_k,local_cache_v]
        return attn_output, new_past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.last = last
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # if last:
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # else:
        #     self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size * 2, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayeremb(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config, last=last)
        self.last = last
        # self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # if self.index!=0:

        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_emb: torch.Tensor,
            hidden_states: torch.Tensor,
            cache_hidden: [List[torch.Tensor]] = [],
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)

        return_hidden = hidden_states

        # cache_hidden.append(hidden_states)

        # Self Attention
        hidden_states, latest_hidden_cache = self.self_attn(
            cache_hidden=cache_hidden,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states


        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, return_hidden)


        return outputs, latest_hidden_cache


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


def process_data(data_chunk):

    token_dict = Counter()
    input_ids = data_chunk["input_ids"]
    loss_mask = data_chunk["loss_mask"]
    for i in range(len(input_ids)):
        ids= input_ids[i][0]
        mask = loss_mask[i][0]
        for j in range(len(ids)):
            if mask[j] == 1:
                token_dict[ids[j]] += 1

    return token_dict


def merge_dicts(dicts):
    """合并多个 Counter 字典"""
    result = Counter()
    for d in dicts:
        result.update(d)
    return result


class Model(nn.Module):
    def __init__(self, config, ds_config, training_config, load_head=False, load_emb=True, path=None, draftpath=None):
        super().__init__() 
        # self.layers = nn.ModuleList(
        #     [LlamaDecoderLayer(config, index=index) for index in range(config.num_hidden_layers)])
        self.train_config = training_config
        # Settng dschf to allow efficient ZeRO-3 usage between hf and ds.
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
        self.midlayer = LlamaDecoderLayeremb(config)
        self.gradient_checkpointing = self.train_config["gradient_checkpoint"]
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.draft_vocab_size = config.draft_vocab_size
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.length = 7
        self.hard_loss_weight = float(self.train_config.get("hard_loss_weight", 0.0))
        self.easy_loss_beta = float(self.train_config.get("easy_loss_beta", 1.0))
        self.last_skd_stats = None
        if not 0.0 <= self.hard_loss_weight <= 1.0:
            raise ValueError("hard_loss_weight must be in [0, 1].")
        if self.easy_loss_beta <= 0.0:
            raise ValueError("easy_loss_beta must be positive.")
        self.target_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        self.target_model.eval()
        self.fc=nn.Linear(self.hidden_size*3, self.hidden_size, bias=False)
        for param in self.target_model.parameters():
            param.requires_grad = False

        if not load_emb:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        else:

            from safetensors import safe_open
            import json
            import os
            from huggingface_hub import snapshot_download

            # If path is a HuggingFace repo ID (not a local dir), resolve to local cache
            if not os.path.isdir(path):
                local_path = snapshot_download(repo_id=path, allow_patterns=["model.safetensors.index.json", "pytorch_model.bin.index.json", "*.safetensors"])
            else:
                local_path = path

            try:
                with open(os.path.join(local_path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(os.path.join(local_path, emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(local_path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(local_path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()
                vocab_size, hidden_dim = tensor.shape
            self.embed_tokens = nn.Embedding(vocab_size, hidden_dim, self.padding_idx, _weight=tensor)

        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)

        for param in self.embed_tokens.parameters():
            param.requires_grad = False

        if draftpath is not None:
            self.load_draft_checkpoint(draftpath)

    def load_draft_checkpoint(self, draftpath):
        checkpoint_path = draftpath
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"draft checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict):
            if "module" in state_dict:
                state_dict = state_dict["module"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        draft_prefixes = ("midlayer.", "norm.", "fc.", "lm_head.", "d2t", "t2d")
        current_state = self.state_dict()
        draft_state = {}
        skipped_keys = []
        for key, value in state_dict.items():
            if not key.startswith(draft_prefixes):
                skipped_keys.append(key)
                continue
            if key not in current_state or current_state[key].shape != value.shape:
                skipped_keys.append(key)
                continue
            draft_state[key] = value

        self.load_state_dict(draft_state, strict=False)
        print(f"Loaded {len(draft_state)} draft checkpoint tensors from {checkpoint_path}")
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} incompatible or non-draft checkpoint tensors.")

    def scandata(self, datapath, tokenizerpath):
        N = self.draft_vocab_size
        if not os.path.exists("cache.pt"):
            tokenizer = AutoTokenizer.from_pretrained(tokenizerpath)
            dataset = load_dataset('json', data_files=datapath)
            dataset = dataset['train']
            # dataset = dataset.select(range(96))
            original_columns1 = dataset.column_names
            num_proc = 8
            max_len = self.train_config["max_len"]


            def preprocess_function(examples):
                new_examples = {
                    # "conversation": [],
                    "input_ids": [],
                    "loss_mask": []
                }
                for i in range(len(examples['id'])):
                    messages = [
                        {"role": "system",
                         "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
                    ]
                    convroles = ["user", "assistant"]
                    roles = {"human": "user", "gpt": "assistant"}
                    source = examples['conversations'][i]
                    if not source:
                        continue
                    if roles[source[0]["from"]] != "user":
                        # Skip the first one if it is not from human
                        source = source[1:]
                    for j, sentence in enumerate(source):
                        role = roles[sentence["from"]]
                        assert role == convroles[j % 2], f"{i}"
                        # if sentence["from"]=="gpt":
                        #     sentence["value"]=" "+sentence["value"]
                        messages.append(
                            {"role": role, "content": sentence["value"]}
                        )
                    conversation = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )

                    if not tokenizer.pad_token_id:
                        tokenizer.pad_token_id = tokenizer.unk_token_id

                    input_ids = tokenizer(
                        conversation,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).input_ids[0]
                    # When construct draft model vocab, 
                    # filter out samples which is longer than max_len,
                    # instead of truncating them.
                    if len(input_ids) > max_len:
                        continue
                    loss_mask = torch.ones_like(input_ids)
                    # print(i)

                    sep = "<|im_start|>assistant"
                    sep2 = "<|im_start|>user"

                    total_len = len(input_ids)
                    turns = conversation.split(sep2)

                    turns[1] = turns[0] + sep2 + turns[1]
                    turns = turns[1:]

                    cur_len = 1
                    loss_mask[:cur_len] = 0
                    for i, turn in enumerate(turns):
                        if turn == "":
                            break
                        turn_len = len(tokenizer(turn).input_ids)

                        parts = turn.split(sep)
                        if len(parts) != 2:
                            break
                        parts[0] += sep
                        # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                        instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                        # Ignore the user instructions
                        if i == 0:
                            loss_mask[cur_len: cur_len + instruction_len - 2] = 0
                        else:
                            loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
                        cur_len += turn_len
                        if i != 0:
                            cur_len += 3
                        # cur_len+=2

                        # if i != 0 and not tokenizer.legacy:
                        #     # The legacy and non-legacy modes handle special tokens differently
                        #     cur_len -= 1

                    loss_mask[cur_len:] = 0

                    # new_examples["conversation"].append(conversation)
                    new_examples["input_ids"].append(input_ids[None, :])
                    new_examples["loss_mask"].append(loss_mask[None, :])

                return new_examples

            dataset = dataset.map(
                preprocess_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=original_columns1,
                load_from_cache_file=True,
                desc="Scanning tokens",
            )
            #dataset.set_format(type="torch")



            num_processes = num_proc
            chunk_size = len(dataset) // num_processes + (len(dataset) % num_processes > 0)
            chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]

            # 创建进程池
            with multiprocessing.Pool(num_processes) as pool:
                # 并行处理数据块
                results = pool.map(process_data, chunks)

            # 合并结果
            token_dict = merge_dicts(results)


            total_frequency = sum(token_dict.values())
            top_N = token_dict.most_common(N)
            top_N_frequency_sum = sum(freq for key, freq in top_N)
            top_N_ratio = top_N_frequency_sum / total_frequency
            print(f"top {N} token frequency ratio: {top_N_ratio:.2%}")
            used_tokens = [key for key, freq in top_N]
            used_tokens.sort()
            d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
            t2d = [i in used_tokens for i in range(self.vocab_size)]
            d2t = torch.tensor(d2t)
            t2d = torch.tensor(t2d)
            cache = {
                "d2t": d2t,
                "t2d": t2d
            }
            torch.save(cache, "cache.pt")
        else:
            cache = torch.load("cache.pt")
            d2t = cache["d2t"]
            t2d = cache["t2d"]
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)
        self.l1smooth = nn.SmoothL1Loss(reduction="none")

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def _sample_token(self, logits, temperature=1.0):
        logits = logits.float()
        if temperature is None or temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        probs = torch.softmax(logits / temperature, dim=-1)
        if not torch.isfinite(probs).all() or probs.sum(dim=-1).min() <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1)

    def _teacher_accepts(self, token, teacher_logits):
        target_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
        teacher_k = int(self.train_config.get("skd_teacher_k", 10))
        teacher_p = float(self.train_config.get("skd_teacher_p", 0.0))
        token = token.view(-1)[0]
        teacher_logits = teacher_logits.float()

        if teacher_k > 0:
            teacher_k = min(teacher_k, teacher_logits.shape[-1])
            topk_ids = torch.topk(teacher_logits, teacher_k, dim=-1).indices
            print(f"token={token.item()} ('{target_tokenizer.decode([token.item()])}'), "
                    f"top5={topk_ids[0, :5].tolist()} "
                    f"({[target_tokenizer.decode([t]) for t in topk_ids[0, :5].tolist()]}), "
                    f"token_rank={(teacher_logits.argsort(descending=True)[0] == token).nonzero().item()}")
            return bool((topk_ids == token).any().item())

        if teacher_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(teacher_logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            keep_mask = cumulative_probs - sorted_probs <= teacher_p
            keep_mask[..., 0] = True
            return bool((sorted_indices[keep_mask] == token).any().item())

        return True

    def _teacher_acceptance_mask(self, candidate_ids, teacher_logits):
        teacher_k = int(self.train_config.get("skd_teacher_k", 10))
        teacher_p = float(self.train_config.get("skd_teacher_p", 0.0))
        candidate_ids = candidate_ids.to(device=teacher_logits.device, dtype=torch.long)
        teacher_logits = teacher_logits.float()

        if teacher_k > 0:
            teacher_k = min(teacher_k, teacher_logits.shape[-1])
            selected_tokens = torch.topk(teacher_logits, teacher_k, dim=-1).indices
            return (selected_tokens == candidate_ids.unsqueeze(-1)).any(dim=-1)

        if teacher_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(teacher_logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            keep_mask = cumulative_probs - sorted_probs <= teacher_p
            keep_mask[..., 0] = True
            selected_tokens = sorted_indices.masked_fill(~keep_mask, -1)
            return (selected_tokens == candidate_ids.unsqueeze(-1)).any(dim=-1)

        return torch.ones(candidate_ids.shape, dtype=torch.bool, device=teacher_logits.device)

    def _debug_teacher_acceptance(self, candidate_ids, teacher_logits, accepted_mask, accepted_prefix_len, confirmed_count):

        tokenizer_name = self.train_config.get("skd_debug_tokenizer", "Qwen/Qwen3-8B")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._skd_debug_tokenizer = tokenizer
        def decode_token(token_id):
            token_id = int(token_id)
            if tokenizer is None:
                return str(token_id)
            try:
                return f"{token_id} ('{tokenizer.decode([token_id])}')"
            except Exception:
                return str(token_id)

        topn = min(int(self.train_config.get("skd_debug_topn", 5)), teacher_logits.shape[-1])
        top_ids = torch.topk(teacher_logits[0], topn, dim=-1).indices
        print(
            f"[skd-debug] accepted_mask={accepted_mask.tolist()}, "
            f"accepted_prefix_len={accepted_prefix_len}, confirmed_count={confirmed_count}"
        )
        for pos in range(candidate_ids.shape[1]):
            token = candidate_ids[0, pos]
            token_logit = teacher_logits[0, pos, token]
            token_rank = int((teacher_logits[0, pos] > token_logit).sum().item())
            top_list = top_ids[pos].tolist()
            decoded_top = [decode_token(token_id) for token_id in top_list]
            status = "ACCEPT" if bool(accepted_mask[pos].item()) else "REJECT"
            print(
                f"[skd-debug] pos={pos} {status} token={decode_token(token.item())}, "
                f"rank={token_rank}, top{topn}={top_list} ({decoded_top})"
            )

    def _draft_to_target_token(self, draft_token):
        draft_token = draft_token.to(dtype=torch.long)
        if self.draft_vocab_size == self.vocab_size:
            return draft_token
        d2t = self.d2t.to(draft_token.device)
        return draft_token + d2t[draft_token]

    def _target_hidden_from_outputs(self, outputs):
        hid_len = len(outputs.hidden_states)
        hidden_states0 = outputs.hidden_states[2]
        hidden_states1 = outputs.hidden_states[hid_len // 2]
        hidden_states2 = outputs.hidden_states[hid_len - 3]
        return torch.cat((hidden_states0, hidden_states1, hidden_states2), dim=-1)

    def _crop_teacher_past(self, past_key_values, keep_len):
        if hasattr(past_key_values, "crop"):
            past_key_values.crop(keep_len)
            return past_key_values

        cropped = []
        for layer_past in past_key_values:
            cropped.append(tuple(past[:, :, :keep_len, :].contiguous() for past in layer_past))
        return tuple(cropped)

    @torch.no_grad()
    def _teacher_forward_with_cache(self, token_ids, past_key_values, output_hidden_states=True, use_cache=True):
        return self.target_model(
            input_ids=token_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

    @torch.no_grad()
    def _eagle_draft_prefill(self, hidden_context, context_ids): # context_ids: generated tokens; hidden_context: target model hidden states
        device = context_ids.device
        draft_input_ids = context_ids[:, 1:]
        if hidden_context.shape[1] != draft_input_ids.shape[1]:
            raise RuntimeError(
                f"EAGLE draft context length mismatch: hidden_context={hidden_context.shape[1]}, "
                f"draft_input_ids={draft_input_ids.shape[1]}."
            )
        inputs_embeds = self.embed_tokens(draft_input_ids).to(hidden_context.dtype)
        seq_len = hidden_context.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
        decoder_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (1, seq_len),
            hidden_context,
            0,
        )
        layer_outputs, cache_hidden = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_context,
            cache_hidden=[[], []],
            attention_mask=decoder_attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=True,
        )
        return layer_outputs[0][:, -1:], cache_hidden # layer_outputs is a tuple

    def _draft_cache_token_count(self, cache_hidden):
        return cache_hidden[0][0].shape[2] + len(cache_hidden[0]) - 1

    def _compact_draft_cache(self, cache_hidden):
        if len(cache_hidden[0]) <= 1:
            return cache_hidden
        return [
            [torch.cat(cache_hidden[0], dim=2).contiguous()],
            [torch.cat(cache_hidden[1], dim=2).contiguous()],
        ]

    @torch.no_grad()
    # Generate hidden states and update cache of newly generated token
    def _eagle_draft_decode_one(self, current_hidden, cache_hidden, token, hidden_state=None):
        device = token.device
        if hidden_state is None:
            hidden_state = current_hidden
        else:
            hidden_state = hidden_state.to(current_hidden.dtype)
        input_embeds = self.embed_tokens(token.view(1, 1).to(device)).to(current_hidden.dtype)
        first_cache_len = cache_hidden[0][0].shape[2]
        decoder_attention_mask = torch.zeros(
            (1, 1, 1, first_cache_len),
            dtype=current_hidden.dtype,
            device=device,
        )
        position_offset = len(cache_hidden[0])
        position_ids = torch.full(
            (1, 1),
            self._draft_cache_token_count(cache_hidden) - position_offset,
            dtype=torch.long,
            device=device,
        )
        layer_outputs, cache_hidden = self.midlayer(
            input_emb=input_embeds,
            hidden_states=hidden_state,
            cache_hidden=cache_hidden,
            attention_mask=decoder_attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=True,
        )
        return layer_outputs[0], cache_hidden

    @torch.no_grad()
    def _eagle_draft_extend_confirmed(self, current_hidden, cache_hidden, hidden_delta, token_delta):
        for idx in range(token_delta.shape[1]):
            current_hidden, cache_hidden = self._eagle_draft_decode_one(
                current_hidden,
                cache_hidden,
                token_delta[:, idx],
                hidden_state=hidden_delta[:, idx:idx + 1],
            )
        return current_hidden, self._compact_draft_cache(cache_hidden)

    @torch.no_grad()
    def _eagle_draft_propose(self, current_hidden, cache_hidden, max_new_tokens):
        if max_new_tokens <= 0:
            return [], None

        student_temperature = float(
            self.train_config.get(
                "skd_student_temperature",
                self.train_config.get("skd_temperature", 1.0),
            )
        )
        proposed_tokens = []
        proposal_hiddens = []

        for draft_idx in range(max_new_tokens):
            proposal_hiddens.append(current_hidden)
            logits = self.lm_head(self.norm(current_hidden)).squeeze(1)
            draft_token = self._sample_token(logits, temperature=student_temperature)
            target_token = self._draft_to_target_token(draft_token)
            proposed_tokens.append(target_token.view(1))

            if draft_idx == max_new_tokens - 1:
                break
            
            current_hidden, cache_hidden = self._eagle_draft_decode_one(
                current_hidden,
                cache_hidden,
                target_token.view(1),
            ) # input single new token, output hidden states for the new token and update cache

        return proposed_tokens, torch.cat(proposal_hiddens, dim=1)

    @torch.no_grad()
    def rollout_skd(self, input_ids, attention_mask, loss_mask):
        device = input_ids.device
        loss_mask = loss_mask.to(device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            attention_mask = attention_mask.to(device)

        gamma = max(int(self.train_config.get("skd_gamma", 5)), 1)
        max_new_tokens = int(self.train_config.get("skd_max_new_tokens", 256))
        # max_new_tokens = 256
        max_len = int(self.train_config.get("max_len", input_ids.shape[1]))
        teacher_temperature = float(
            self.train_config.get(
                "skd_teacher_temperature",
                self.train_config.get("skd_temperature", 1.0),
            )
        )
        if not bool(self.train_config.get("skd_seed_with_teacher", True)):
            raise ValueError("EAGLE-compatible SKD rollout requires one initial teacher seed.")
        eos_token_id = getattr(self.target_model.config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_ids = set(eos_token_id)
        elif eos_token_id is None:
            eos_token_ids = set()
        else:
            eos_token_ids = {int(eos_token_id)}

        batch_input_ids = []
        batch_attention_mask = []
        batch_loss_mask = []
        total_accepted = 0
        total_checked = 0
        total_corrections = 0
        rollout_lengths = []

        for batch_idx in range(input_ids.shape[0]):
            valid_len = int(attention_mask[batch_idx].long().sum().item())
            sample_input_ids = input_ids[batch_idx, :valid_len]
            sample_loss_mask = loss_mask[batch_idx, :valid_len]
            loss_positions = torch.where(sample_loss_mask > 0)[0]

            if len(loss_positions) == 0:
                new_input_ids = sample_input_ids
                new_attention_mask = torch.ones_like(new_input_ids, device=device)
                new_loss_mask = torch.zeros_like(new_input_ids, device=device)
                batch_input_ids.append(new_input_ids)
                batch_attention_mask.append(new_attention_mask)
                batch_loss_mask.append(new_loss_mask)
                rollout_lengths.append(0)
                continue

            prompt_len = max(int(loss_positions[0].item()), 1)
            target_new_tokens = min(
                int(sample_loss_mask.sum().item()),
                max_new_tokens,
                max(max_len - prompt_len, 0),
            )
            prefix_ids = sample_input_ids[:prompt_len].view(1, -1)
            generated = prefix_ids.clone()
            generated_loss_mask = [0] * prompt_len
            generated_new_tokens = 0
            hidden_context = None
            teacher_past = None
            draft_current_hidden = None
            draft_cache_hidden = None

            if target_new_tokens > 0:
                prefix_attention_mask = torch.ones_like(generated, device=device)
                teacher_outputs = self.target_model(
                    input_ids=generated,
                    attention_mask=prefix_attention_mask,
                    use_cache=True,
                    output_hidden_states=True,
                )
                teacher_logits = teacher_outputs.logits[:, -1, :]
                seed_token = self._sample_token(teacher_logits, temperature=teacher_temperature)
                hidden_context = self.fc(self._target_hidden_from_outputs(teacher_outputs))
                teacher_past = teacher_outputs.past_key_values
                if teacher_past is None:
                    raise RuntimeError("target_model did not return past_key_values; cached SKD rollout requires use_cache=True.")
                generated = torch.cat((generated, seed_token.to(device)), dim=1)
                generated_loss_mask.append(1)
                generated_new_tokens += 1
                draft_current_hidden, draft_cache_hidden = self._eagle_draft_prefill(hidden_context, generated)

                if int(seed_token.item()) in eos_token_ids:
                    target_new_tokens = generated_new_tokens

            while generated_new_tokens < target_new_tokens and generated.shape[1] < max_len:
                remaining_tokens = min(
                    gamma,
                    target_new_tokens - generated_new_tokens,
                    max_len - generated.shape[1],
                )
                if remaining_tokens <= 0:
                    continue

                draft_tokens, proposal_hiddens = self._eagle_draft_propose(
                    draft_current_hidden,
                    draft_cache_hidden,
                    remaining_tokens,
                )
                if not draft_tokens:
                    continue

                candidate_ids = torch.cat([token.view(1, 1).to(device) for token in draft_tokens], dim=1) # shape (1, len_draft_tokens)
                verify_input_ids = torch.cat((generated[:, -1:], candidate_ids), dim=1)
                
                verify_outputs = self.target_model(
                    input_ids=verify_input_ids,
                    past_key_values=teacher_past,
                    use_cache=True,
                    output_hidden_states=False,
                )
                verify_logits = verify_outputs.logits[:, :-1, :]
                verify_past = verify_outputs.past_key_values

                accepted_mask = self._teacher_acceptance_mask(candidate_ids, verify_logits).squeeze(0)
                if bool(accepted_mask.all().item()):
                    accepted_prefix_len = candidate_ids.shape[1]
                    rejected = False
                else:
                    accepted_prefix_len = int((~accepted_mask).nonzero(as_tuple=False)[0].item())
                    rejected = True

                confirmed_count = min(
                    accepted_prefix_len,
                    target_new_tokens - generated_new_tokens,
                    max_len - generated.shape[1],
                )
                hit_eos = False
                if confirmed_count > 0 and eos_token_ids:
                    eos_mask = torch.zeros((confirmed_count,), dtype=torch.bool, device=device)
                    for eos_id in eos_token_ids:
                        eos_mask |= candidate_ids[0, :confirmed_count].eq(eos_id)
                    if bool(eos_mask.any().item()):
                        confirmed_count = int(eos_mask.nonzero(as_tuple=False)[0].item()) + 1
                        hit_eos = True

                self._debug_teacher_acceptance(
                    candidate_ids,
                    verify_logits,
                    accepted_mask,
                    accepted_prefix_len,
                    confirmed_count,
                )

                if confirmed_count > 0:
                    accepted_tokens = candidate_ids[:, :confirmed_count]
                    generated = torch.cat((generated, accepted_tokens), dim=1)
                    generated_loss_mask.extend([1] * confirmed_count)
                    generated_new_tokens += confirmed_count
                    total_accepted += confirmed_count
                total_checked += confirmed_count

                replacement_token = None
                can_replace = (
                    rejected
                    and confirmed_count == accepted_prefix_len
                    and not hit_eos
                    and generated_new_tokens < target_new_tokens
                    and generated.shape[1] < max_len
                    and int(generated[0, -1].item()) not in eos_token_ids
                )
                if can_replace:
                    total_checked += 1
                    replacement_token = self._sample_token(
                        verify_logits[:, accepted_prefix_len, :],
                        temperature=teacher_temperature,
                    )
                    generated = torch.cat((generated, replacement_token.to(device)), dim=1)
                    generated_loss_mask.append(1)
                    generated_new_tokens += 1
                    total_corrections += 1

                last_token = int(generated[0, -1].item())
                if replacement_token is None and confirmed_count > 0:
                    teacher_past = self._crop_teacher_past(verify_past, generated.shape[1] - 1)
                    hidden_delta = proposal_hiddens[:, :confirmed_count]
                    token_delta = candidate_ids[:, :confirmed_count]
                    draft_current_hidden, draft_cache_hidden = self._eagle_draft_extend_confirmed(
                        draft_current_hidden,
                        draft_cache_hidden,
                        hidden_delta,
                        token_delta,
                    )
                elif replacement_token is not None:
                    teacher_past = self._crop_teacher_past(verify_past, generated.shape[1] - 1)
                    hidden_delta = proposal_hiddens[:, :confirmed_count + 1]
                    token_delta = torch.cat(
                        (candidate_ids[:, :confirmed_count], replacement_token.to(device)),
                        dim=1,
                    )
                    draft_current_hidden, draft_cache_hidden = self._eagle_draft_extend_confirmed(
                        draft_current_hidden,
                        draft_cache_hidden,
                        hidden_delta,
                        token_delta,
                    )
                if last_token in eos_token_ids:
                    break

            new_input_ids = generated.squeeze(0)
            new_attention_mask = torch.ones_like(new_input_ids, device=device)
            new_loss_mask = torch.tensor(generated_loss_mask, dtype=loss_mask.dtype, device=device)
            batch_input_ids.append(new_input_ids)
            batch_attention_mask.append(new_attention_mask)
            batch_loss_mask.append(new_loss_mask)
            rollout_lengths.append(int(new_loss_mask.sum().item()))

        padded_len = max(tensor.shape[0] for tensor in batch_input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        padded_loss_mask = []
        pad_token_id = self.padding_idx if self.padding_idx is not None else 0
        for ids, attn, mask in zip(batch_input_ids, batch_attention_mask, batch_loss_mask):
            pad_len = padded_len - ids.shape[0]
            if pad_len > 0:
                ids = torch.cat((ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype, device=device)), dim=0)
                attn = torch.cat((attn, torch.zeros(pad_len, dtype=attn.dtype, device=device)), dim=0)
                mask = torch.cat((mask, torch.zeros(pad_len, dtype=mask.dtype, device=device)), dim=0)
            padded_input_ids.append(ids)
            padded_attention_mask.append(attn)
            padded_loss_mask.append(mask)

        self.last_skd_stats = {
            "accept_rate": total_accepted / (total_checked + 1e-6),
            "correction_rate": total_corrections / (total_checked + 1e-6),
            "avg_rollout_len": sum(rollout_lengths) / (len(rollout_lengths) + 1e-6),
        }
        return (
            torch.stack(padded_input_ids, dim=0),
            torch.stack(padded_attention_mask, dim=0),
            torch.stack(padded_loss_mask, dim=0),
        )

    @torch.no_grad()
    def dataprepare(self, input_ids, attention_mask, loss_mask):
        device = input_ids.device
        outs = self.target_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hid_len = len(outs.hidden_states)
        hidden_states0 = outs.hidden_states[2]
        hidden_states1 = outs.hidden_states[hid_len // 2]
        hidden_states2 = outs.hidden_states[hid_len - 3]
        hidden_states=torch.cat((hidden_states0,hidden_states1,hidden_states2),dim=-1)
        # hidden_states=torch.cat((hidden_states0,hidden_states1),dim=-1)
        target = outs.logits
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)

        if target is not None:
            target = target.to(device)
            loss_mask = loss_mask[..., None]
            loss_mask = loss_mask.to(device)

        return hidden_states, target, loss_mask, input_ids

    def forward(
            self,
            # hidden_states,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            loss_mask: Optional[torch.Tensor] = None,
            skd_rollout: bool = False,

    ):
        if skd_rollout:
            input_ids, attention_mask, loss_mask = self.rollout_skd(input_ids, attention_mask, loss_mask)
        else:
            self.last_skd_stats = None

        hidden_states, target, loss_mask, input_ids = self.dataprepare(input_ids, attention_mask, loss_mask)

        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # with torch.no_grad():
        #     inputs_embeds = self.embed_tokens(input_ids)
        #     inputs_embeds = inputs_embeds.detach()

        if self.training and self.gradient_checkpointing and not hidden_states.requires_grad:
            hidden_states.requires_grad = True

        hidden_states=self.fc(hidden_states)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        plosses = []
        vlosses = []
        acces = []
        cache_hidden = [[], []]

        for idx in range(self.length):
            last = idx == self.length - 1
            inputs_embeds = self.embed_tokens(input_ids)
            if self.training and self.gradient_checkpointing and not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad = True
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, None, output_attentions)

                    return custom_forward

                layer_outputs, cache_hidden = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.midlayer),
                    inputs_embeds,
                    hidden_states,
                    cache_hidden,
                    attention_mask,
                    position_ids,
                )
            else:

                layer_outputs, cache_hidden = self.midlayer(
                    input_emb=inputs_embeds,
                    hidden_states=hidden_states,
                    cache_hidden=cache_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=output_attentions,
                    use_cache=True,
                )

            hidden_states_out = layer_outputs[0]
            # cache_hidden.append(layer_outputs[1])
            # kv_cahce = layer_outputs[-1]

            with torch.no_grad():
                # hidden_states_target = padding(hidden_states, left=False)
                target_head = target
                target_max_token = target_head.argmax(-1)
                # Move d2t to the same device as target_max_token
                self.t2d = self.t2d.to(target_max_token.device)
                target_mask = self.t2d[target_max_token]
                target_mask = target_mask[..., None].int()
                position_mask = target_mask * loss_mask
                target_head = target_head[..., self.t2d]
                target_head = target_head.float()
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()



            hidden_states = hidden_states_out

            hidden_states_out = self.norm(hidden_states_out)

            logits = self.lm_head(hidden_states_out)
            logits = logits.float()
            out_logp = nn.LogSoftmax(dim=2)(logits)
            token_ce = -(target_p * out_logp).sum(dim=2)
            easy_token_score = torch.exp(-self.easy_loss_beta * token_ce)
            easy_token_loss = 1.0 - easy_token_score

            token_loss = self.hard_loss_weight * token_ce + (1.0 - self.hard_loss_weight) * easy_token_loss

            loss = -torch.sum(position_mask * token_loss, 2).mean()
            plosses.append(loss)
            with torch.no_grad():
                acces.append(((logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)).sum().item() / (
                        loss_mask.sum().item() + 1e-6))

            if not last:
                input_ids = padding(input_ids, left=False)
                target = padding(target, left=False)
                loss_mask = padding(loss_mask, left=False)



        return plosses, vlosses, acces




