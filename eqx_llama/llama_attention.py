import math
from typing import Literal

import chex
import equinox as eqx
import jax
import jax.experimental.pallas.ops.gpu.attention as pla
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .internals import mha
from .utils import (
    KVCache,
    LLaMAConfig,
    RMSLayerNorm,
    apply_rotary_embeddings,
    init_weights,
    safe_concat,
)

# TODO make sure to pad inputs to pallas attention to the nearest power of 2


def cache_get(cache: KVCache | None, cache_key: str):
    if cache is None:
        return None, None, 0
    ks, vs = cache.get(cache_key)
    context_len = ks.shape[0] if ks is not None else 0
    return ks, vs, context_len


def cache_set(cache: KVCache | None, cache_key: str, ks: Array, vs: Array):
    if cache is None:
        return None
    return cache.set(cache_key, ks, vs)


class _AttentionWeights(eqx.Module):
    wq: Array
    wk: Array
    wv: Array
    wo: Array

    def __init__(self, wq, wk, wv, wo):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo


class AttentionModule(eqx.Module):
    norm: RMSLayerNorm
    weights: _AttentionWeights

    layer_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    cache_key: str = eqx.field(static=True)

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: PRNGKeyArray,
        dtype: jax.typing.DTypeLike = "float32",
    ):
        assert (
            config.attention_num_heads * config.attention_head_dim == config.layer_dim
        )
        k1, k2, k3, k4, key = jax.random.split(key, 5)

        self.layer_dim = config.layer_dim
        self.num_heads = config.attention_num_heads
        self.head_dim = config.attention_head_dim

        self.norm = RMSLayerNorm(config.layer_dim)
        self.weights = _AttentionWeights(
            init_weights((config.layer_dim, self.num_heads, self.head_dim), k1, dtype),
            init_weights((config.layer_dim, self.num_heads, self.head_dim), k2, dtype),
            init_weights((config.layer_dim, self.num_heads, self.head_dim), k3, dtype),
            init_weights((config.layer_dim, self.num_heads, self.head_dim), k4, dtype),
        )

        self.cache_key = str(id(self))

    def _compute_embeddings(
        self,
        xs: Float[Array, "seq_len layer_dim"],
        start_index: int = 0,
    ) -> tuple[
        Float[Array, "seq_len num_heads head_dim"],
        Float[Array, "seq_len num_heads head_dim"],
        Float[Array, "seq_len num_heads head_dim"],
    ]:
        apply_rope = jax.vmap(
            lambda h: apply_rotary_embeddings(h, start_index), in_axes=1, out_axes=1
        )
        qs = apply_rope(jnp.einsum("sd,dnh->snh", xs, self.weights.wq))
        ks = apply_rope(jnp.einsum("sd,dnh->snh", xs, self.weights.wk))
        vs = jnp.einsum("sd,dnh->snh", xs, self.weights.wv)
        return qs, ks, vs

    def __call__(
        self,
        xs: Float[Array, "seq_len layer_dim"],
        cache: KVCache | None,
        attn_implementation: Literal["pallas", "regular"] = "regular",
    ) -> tuple[Float[Array, "seq_len layer_dim"], KVCache | None]:
        seq_len = xs.shape[0]

        old_ks, old_vs, context_len = cache_get(cache, self.cache_key)
        new_qs, new_ks, new_vs = self._compute_embeddings(self.norm(xs), context_len)
        ks, vs = safe_concat(old_ks, new_ks), safe_concat(old_vs, new_vs)
        cache = cache_set(cache, self.cache_key, ks, vs)

        attn_out = compute_self_attention(new_qs, ks, vs, attn_implementation)
        out = jnp.einsum("snh,dnh->sd", attn_out, self.weights.wo)

        if old_ks is not None:
            chex.assert_shape(
                [old_ks, old_vs], (context_len, self.num_heads, self.head_dim)
            )
        chex.assert_shape(
            [new_qs, new_ks, new_vs], (seq_len, self.num_heads, self.head_dim)
        )
        chex.assert_shape([xs, out], (seq_len, self.layer_dim))

        return out, cache


def compute_self_attention(
    qs: Float[Array, "seq_len num_heads head_dim"],
    ks: Float[Array, "context_len+seq_len num_heads head_dim"],
    vs: Float[Array, "context_len+seq_len num_heads head_dim"],
    attn_implementation: Literal["pallas", "regular"] = "regular",
    **kwargs,
) -> Float[Array, "seq_len num_heads head_dim"]:
    if attn_implementation == "pallas":
        return pla.mha(
            qs[None, ...],
            ks[None, ...],
            vs[None, ...],
            None,
            sm_scale=1 / math.sqrt(qs.shape[2]),
            causal=True,
            **kwargs,
        )[0, ...]

    if attn_implementation == "regular":
        return mha(
            qs[None, ...],
            ks[None, ...],
            vs[None, ...],
            sm_scale=1 / math.sqrt(qs.shape[2]),
            causal=True,
        )[0, ...]

    raise ValueError(f"Unexpected attention implementation '{attn_implementation}'")
