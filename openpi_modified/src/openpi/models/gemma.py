# Copyright 2024 Big Vision Authors.
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

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias, Any

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)
    # Gemma 3 specific: enable q_norm, k_norm, post_attention_norm, post_ffw_norm
    use_gemma3: bool = False


Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora", "gemma_270m", "gemma_3_270m"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={
                "attn": lora.LoRAConfig(rank=16, alpha=16.0),
                "ffn": lora.LoRAConfig(rank=16, alpha=16.0),
            },
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={
                "attn": lora.LoRAConfig(rank=32, alpha=32.0),
                "ffn": lora.LoRAConfig(rank=32, alpha=32.0),
            },
        )
    if variant == "gemma_270m":
        # Gemma-3-270M from HuggingFace
        return Config(
            width=640,
            depth=18,
            mlp_dim=2048,
            num_heads=4,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_3_270m":
        # Gemma-3-270M with full Gemma 3 architecture support
        # Includes q_norm, k_norm, post_attention_norm, post_ffw_norm
        return Config(
            width=640,
            depth=18,
            mlp_dim=2048,
            num_heads=4,
            num_kv_heads=1,
            head_dim=256,
            use_gemma3=True,
        )
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(
            jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True
        )  # compute variance in float32
        normed_inputs = jnp.asarray(
            x * jnp.reciprocal(jnp.sqrt(var + 1e-06))
        )  # compute normalization in float32
        if cond is None:
            # regular RMSNorm
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (
                1 + scale
            )  # scale by learned parameter in float32 (matches Flax implementation)
            return normed_inputs.astype(dtype), None  # return in original dtype

        # adaptive RMSNorm
        modulation = nn.Dense(
            x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype
        )(cond)
        if modulation.ndim == x.ndim:
            # RTC case: different timestamp for each token
            modulation_reshaped = modulation
        else:
            modulation_reshaped = modulation[:, None, :]

        scale, shift, gate = jnp.split(modulation_reshaped, 3, axis=-1)
        normed_inputs = (
            normed_inputs * (1 + scale) + shift
        )  # scale and shift in float32
        return normed_inputs.astype(dtype), gate


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    def _init_kv_cache(self, k, v, cache_size):
        """Initialize KV cache"""
        prefill_len = k.shape[1]
        # Ensure cache_size is concrete
        # cache_size = attn_mask.shape[-1]
        pad_width = ((0, 0), (0, cache_size - prefill_len), (0, 0), (0, 0))
        cache_dtype = k.dtype
        k_cache = jnp.pad(k.astype(cache_dtype), pad_width)
        v_cache = jnp.pad(v.astype(cache_dtype), pad_width)
        idx = jnp.zeros((k.shape[0],), dtype=jnp.int32) + prefill_len
        return idx, k_cache, v_cache

    def _update_kv_cache(self, k, v, idx, k_cache, v_cache):
        """Update KV cache with new values"""
        assert k.shape[1] == 1, "Only support kv-cache updates of length 1"
        indices = (0, idx[0], 0, 0)
        cache_dtype = k.dtype
        k_new = jax.lax.dynamic_update_slice(k_cache, k.astype(cache_dtype), indices)
        v_new = jax.lax.dynamic_update_slice(v_cache, v.astype(cache_dtype), indices)
        idx_new = idx + 1
        return idx_new, k_new, v_new

    def fill_kv_cache_update(self, k, v, attn_mask, kv_cache):
        if kv_cache is None:
            idx, k_cache, v_cache = self._init_kv_cache(k, v, attn_mask)
            return (idx, k_cache, v_cache)
        idx, k_cache, v_cache = kv_cache
        idx, k_cache, v_cache = self._update_kv_cache(k, v, idx, k_cache, v_cache)
        return (idx, k_cache, v_cache)

    def kv_cache_update(self, k, v, kv_cache):
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)
        idx = jnp.zeros((k.shape[0],), dtype=jnp.int32)
        kv_cache = (idx, k, v)
        return kv_cache

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache, fill_kv_cache=False):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(
            config.head_dim == self.configs[0].head_dim for config in self.configs
        )
        assert all(
            config.num_heads == self.configs[0].num_heads for config in self.configs
        )
        assert all(
            config.num_kv_heads == self.configs[0].num_kv_heads
            for config in self.configs
        )

        dtype = next(
            x.dtype for x in xs if x is not None
        )  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(
                        in_axis=-2, out_axis=-1, batch_axis=(0, 1)
                    ),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(
                        in_axis=-2, out_axis=-1, batch_axis=(0,)
                    ),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(
                        in_axis=-2, out_axis=-1, batch_axis=(0, 1)
                    ),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        # Gemma 3: apply q_norm and k_norm (RMS normalization)
        if any(config.use_gemma3 for config in self.configs if config.use_gemma3):
            q_norm = self.param("q_norm", nn.initializers.ones_init(), (self.configs[0].head_dim,))
            k_norm = self.param("k_norm", nn.initializers.ones_init(), (self.configs[0].head_dim,))
            # Apply RMS norm to the last dimension (head_dim)
            q_var = jnp.mean(jnp.square(q.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6
            q = (q * jnp.reciprocal(jnp.sqrt(q_var)) * q_norm).astype(dtype)
            k_var = jnp.mean(jnp.square(k.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6
            k = (k * jnp.reciprocal(jnp.sqrt(k_var)) * k_norm).astype(dtype)

        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if fill_kv_cache:
            if kv_cache is None:
                idx, k_cache, v_cache = self._init_kv_cache(k, v, attn_mask.shape[-1])
            else:
                idx, k_cache, v_cache = kv_cache
                idx, k_cache, v_cache = self._update_kv_cache(k, v, idx, k_cache, v_cache)
            kv_cache = (idx, k_cache, v_cache)
            k, v = k_cache, v_cache
        else:
            if kv_cache is not None:
                cache_k, cache_v = kv_cache
                k = jnp.concatenate([cache_k, k], axis=1)
                v = jnp.concatenate([cache_v, v], axis=1)
            kv_cache = (k, v)
        # kv_cache = jax.lax.cond(
        #     fill_kv_cache,  # 允许追踪布尔数组作为条件
        #     self.fill_kv_cache_update,  # True分支
        #     self.kv_cache_update,   # False分支
        #     k, v, attn_mask, kv_cache  # 传递给分支的参数
        # )

        q = einops.rearrange(
            q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads
        )
        logits = jnp.einsum(
            "BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32
        )

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, kv_cache


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    configs: tuple[Config, ...]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(
        self, 
        xs, 
        kv_cache,
        positions, 
        attn_mask, 
        adarms_cond, 
        deterministic=True, 
        fill_kv_cache=False,
    ):  # noqa: FBT002
        xs = sharding.activation_sharding_constraint(xs)
        drop = (
            nn.Dropout(self.dropout, self.dropout_bdims)
            if self.dropout
            else lambda x, _: x
        )

        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(
                    x, adarms_cond[i]
                )  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache, fill_kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)

        # Gemma 3: apply post_attention_norm (after attention, before residual)
        if any(config.use_gemma3 for config in self.configs if config.use_gemma3):
            post_attn_normed = []
            for i, y in enumerate(post_attn):
                if y is not None:
                    y_norm, _ = RMSNorm(name=_name("post_attention_norm", i))(y, None)
                    post_attn_normed.append(y_norm)
                else:
                    post_attn_normed.append(None)
            post_attn = post_attn_normed

        xs = [
            _gated_residual(x, y, gate)
            for x, y, gate in zip(xs, post_attn, gates, strict=True)
        ]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(
                    x, adarms_cond[i]
                )  # noqa: PLW2901
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)

        # Gemma 3: apply post_ffw_norm (after FFN, before residual)
        if any(config.use_gemma3 for config in self.configs if config.use_gemma3):
            out_normed = []
            for i, y in enumerate(out):
                if y is not None:
                    y_norm, _ = RMSNorm(name=_name("post_ffw_norm", i))(y, None)
                    out_normed.append(y_norm)
                else:
                    out_normed.append(None)
            out = out_normed

        xs = [
            _gated_residual(x, y, gate)
            for x, y, gate in zip(xs, out, gates, strict=True)
        ]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


KVCache: TypeAlias = tuple[
    at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]
]


@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str
    vocab_size: int | None = None  # Override vocabulary size (None uses PALIGEMMA_VOCAB_SIZE)

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False

    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        # Use provided vocab_size or fall back to default
        vocab_size = self.vocab_size if self.vocab_size is not None else PALIGEMMA_VOCAB_SIZE

        self.embedder = Embedder(
            vocab_size=vocab_size,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5, 7),  # 0=self, 6=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),  # 0=kv_cache, 1=positions, 2=mask, 3=adarms_cond, 4=deterministic
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        self.final_norms = [
            RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))
        ]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    # @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _t _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
        fill_kv_cache: bool = False,
        pre_logits: at.Float[at.Array, "b _t _d"] | None = None,
        autoregressive_mode: bool = False,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        if pre_logits is not None:
            logits_first = self.embedder.decode(pre_logits)
            return logits_first, kv_cache
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, kv_cache = self.layers(
            embedded, kv_cache, positions, mask, adarms_cond, deterministic, fill_kv_cache
        )
        assert all(
            e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None
        )
        outs = [
            f(e, a)[0] if e is not None else e
            for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ]
        if autoregressive_mode and outs[0] is not None:
            pre_logits_first = outs[0]
            if pre_logits_first.shape[1] > 1:
                pre_logits_first = pre_logits_first[:, -1:]
            logits_first = self.embedder.decode(pre_logits_first)
            outs[0] = logits_first
        return outs, kv_cache

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[
                jnp.zeros((1, c.width)) if u else None
                for u, c in zip(use_adarms, self.configs, strict=True)
            ],
        )


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(
        x.shape[-1] // 2, dtype=jnp.float32
    )
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"


def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate
