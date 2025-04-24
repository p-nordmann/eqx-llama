from typing import Literal

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .kv_cache import KVCache
from .llama_config import LLaMAConfig
from .normalization import RMSLayerNorm


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
        new_state = cache.set(id(self), ks, vs)

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

        return jax.vmap(self.linear_o)(jax.lax.collapse(attention_out, 1, 3)), new_state


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


# def rotary_kernel(x: Float32[Array, " 2"], m_theta: float) -> Float32[Array, " 2"]:
def rotary_kernel(x, m_theta):
    """The core operation of rotary embeddings acts over two dimensions."""
    theta_kernel = jnp.array(
        [
            [jnp.cos(m_theta), -jnp.sin(m_theta)],
            [jnp.sin(m_theta), jnp.cos(m_theta)],
        ]
    )
    return theta_kernel @ x


# def generalized_rotary_kernel(
#     x: Float32[Array, " size"], m: Float32, thetas: Float32[Array, " half_size"]
# ) -> Float32[Array, " half_size"]:
def generalized_rotary_kernel(x, m, thetas):
    """Applies the rotary kernel along a vector of even dimension.

    Thetas must be provided.
    """
    pairs_of_xi = jnp.reshape(
        x,
        shape=(-1, 2),
        order="C",  # Order is critical to be consistent with the original LLaMAs!
    )
    pairs_of_embeddings = jax.vmap(rotary_kernel)(pairs_of_xi, m * thetas)
    return jax.lax.collapse(pairs_of_embeddings, 0, 2)


def apply_rotary_embeddings(
    xs: Float[Array, " seq_len size"],
    start_index: int = 0,
    *,
    theta_base: float = 1e4,
    dtype=jnp.float16,
) -> Float[Array, " seq_len size"]:
    """Applies the rotary kernel through a full sequence with even dimension."""
    half_dim = xs.shape[1] // 2
    ms = jnp.arange(start_index, xs.shape[0] + start_index)
    thetas = theta_base ** (-jnp.arange(0, half_dim, dtype=dtype) / half_dim)
    return jax.vmap(generalized_rotary_kernel, in_axes=[0, 0, None])(xs, ms, thetas)
