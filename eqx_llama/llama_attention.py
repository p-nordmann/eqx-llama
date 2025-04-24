from typing import Literal

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .normalization import RMSLayerNorm
from .utils import KVCache, LLaMAConfig, apply_rotary_embeddings


class AttentionModule(eqx.Module):
    norm: RMSLayerNorm
    rope_embeddings: eqx.nn.RotaryPositionalEmbedding
    linear_q: eqx.nn.Linear
    linear_k: eqx.nn.Linear
    linear_v: eqx.nn.Linear
    linear_o: eqx.nn.Linear

    num_attention_heads: int = eqx.field(static=True)
    size_attention_heads: int = eqx.field(static=True)
    attn_implementation: Literal["xla", "cudnn"] = eqx.field(static=True)

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: PRNGKeyArray,
        attn_implementation: Literal["xla", "cudnn"] = "xla",
    ):
        assert (
            config.num_attention_heads * config.size_attention_heads
            == config.size_layer
        )

        self.num_attention_heads = config.num_attention_heads
        self.size_attention_heads = config.size_attention_heads
        self.attn_implementation = attn_implementation

        self.norm = RMSLayerNorm(config.size_layer)

        self.rope_embeddings = eqx.nn.RotaryPositionalEmbedding(
            config.size_attention_heads, 10_000
        )

        key_linear, key = jax.random.split(key)
        self.linear_q = eqx.nn.Linear(
            config.size_layer,
            config.size_layer,
            use_bias=False,
            key=key_linear,
        )

        key_linear, key = jax.random.split(key)
        self.linear_k = eqx.nn.Linear(
            config.size_layer,
            config.size_layer,
            use_bias=False,
            key=key_linear,
        )

        key_linear, key = jax.random.split(key)
        self.linear_v = eqx.nn.Linear(
            config.size_layer,
            config.size_layer,
            use_bias=False,
            key=key_linear,
        )

        key_linear, key = jax.random.split(key)
        self.linear_o = eqx.nn.Linear(
            config.size_layer,
            config.size_layer,
            use_bias=False,
            key=key_linear,
        )

    def _compute_embeddings(
        self,
        xs: Float[Array, " seq_len size_layer"],
        linear: eqx.nn.Linear,
        use_position_embeddings: bool = False,
        start_index: int = 0,
    ) -> Float[Array, " seq_len num_heads size_heads"]:
        projected_xs = jax.vmap(linear)(xs)
        hs = jnp.reshape(
            projected_xs,
            shape=(-1, self.num_attention_heads, self.size_attention_heads),
        )
        if not use_position_embeddings:
            return hs
        return jax.vmap(apply_rotary_embeddings, in_axes=(1, None), out_axes=1)(
            hs, start_index
        )

    def __call__(
        self,
        xs: Float[Array, " seq_len size_layer"],
        cache: KVCache,
    ) -> tuple[Float[Array, " seq_len size_layer"], KVCache]:
        seq_len = xs.shape[0]
        xs_normalized = jax.vmap(self.norm)(xs)

        # Retrieve cached keys and values.
        old_ks, old_vs = cache.get(id(self))
        if old_ks is None:
            old_ks = jnp.empty((0, self.num_attention_heads, self.size_attention_heads))
        if old_vs is None:
            old_vs = jnp.empty((0, self.num_attention_heads, self.size_attention_heads))
        context_len = old_ks.shape[0]

        chex.assert_equal_shape([old_ks, old_vs])
        chex.assert_axis_dimension(old_ks, 1, self.num_attention_heads)
        chex.assert_axis_dimension(old_ks, 2, self.size_attention_heads)

        # Compute new keys and values (using updated indices for RoPE).
        new_ks = self._compute_embeddings(
            xs_normalized,
            self.linear_k,
            use_position_embeddings=True,
            start_index=context_len,
        )
        new_vs = self._compute_embeddings(xs_normalized, self.linear_v)

        chex.assert_equal_shape([new_ks, new_vs])
        chex.assert_axis_dimension(new_ks, 0, seq_len)
        chex.assert_axis_dimension(new_ks, 1, self.num_attention_heads)
        chex.assert_axis_dimension(new_ks, 2, self.size_attention_heads)

        # Concat full keys and values and update state.
        ks = jnp.concat([old_ks, new_ks], axis=0)
        vs = jnp.concat([old_vs, new_vs], axis=0)
        new_cache = cache.set(id(self), ks, vs)

        # Compute queries (using updated indices for RoPE).
        new_qs = self._compute_embeddings(
            xs_normalized,
            self.linear_q,
            use_position_embeddings=True,
            start_index=context_len,
        )

        chex.assert_equal_shape([new_qs, new_ks])

        # Compute attention and return.
        attention_out = compute_self_attention(
            new_qs, ks, vs, attn_implementation=self.attn_implementation
        )

        chex.assert_equal_shape([attention_out, new_qs])

        return jax.vmap(self.linear_o)(jax.lax.collapse(attention_out, 1, 3)), new_cache


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
