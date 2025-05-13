from typing import Literal

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .utils import (
    KVCache,
    LLaMAConfig,
    RMSLayerNorm,
    apply_rotary_embeddings,
    init_weights,
    safe_concat,
)


class _AttentionWeights(eqx.Module):
    wq: Array
    wk: Array
    wv: Array
    wo: Array


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

        self.cache_key = id(self)

    def _compute_embeddings(
        self,
        xs: Float[Array, " seq_len layer_dim"],
        start_index: int = 0,
    ) -> tuple[
        Float[Array, " seq_len num_heads head_dim"],
        Float[Array, " seq_len num_heads head_dim"],
        Float[Array, " seq_len num_heads head_dim"],
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
        xs: Float[Array, " seq_len layer_dim"],
        cache: KVCache,
        attn_implementation: Literal["xla", "cudnn"] = "xla",
    ) -> tuple[Float[Array, " seq_len layer_dim"], KVCache]:
        seq_len = xs.shape[0]

        old_ks, old_vs = cache.get(self.cache_key)
        context_len = old_ks.shape[0] if old_ks is not None else 0

        xs_normalized = jax.vmap(self.norm)(xs)
        new_qs, new_ks, new_vs = self._compute_embeddings(
            xs_normalized, start_index=context_len
        )

        ks, vs = safe_concat(old_ks, new_ks), safe_concat(old_vs, new_vs)
        cache = cache.set(self.cache_key, ks, vs)

        attention_out = compute_self_attention(
            new_qs, ks, vs, attn_implementation=attn_implementation
        )
        out = jnp.einsum("snh,dnh->sd", attention_out, self.weights.wo)

        if old_ks is not None:
            chex.assert_shape(
                [old_ks, old_vs], (context_len, self.num_heads, self.head_dim)
            )
        chex.assert_shape(
            [new_qs, new_ks, new_vs], (seq_len, self.num_heads, self.head_dim)
        )
        chex.assert_shape([xs, xs_normalized, out], (seq_len, self.layer_dim))

        return out, cache


def compute_self_attention(
    qs: Float[Array, " seq_len num_heads head_dim"],
    ks: Float[Array, " context_len+seq_len num_heads head_dim"],
    vs: Float[Array, " context_len+seq_len num_heads head_dim"],
    *,
    attn_implementation: Literal["xla", "cudnn"] = "xla",
) -> Float[Array, " seq_len num_heads head_dim"]:
    return jax.nn.dot_product_attention(
        qs, ks, vs, is_causal=True, implementation=attn_implementation
    )
