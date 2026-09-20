from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class LLaMAConfig(NamedTuple):
    num_layers: int
    vocab_size: int
    layer_dim: int
    attention_num_heads: int
    attention_head_dim: int
    feed_forward_dim: int


class RMSLayerNorm(eqx.Module):
    """Similar to layer normalization, without the mean estimate.

    Known to give similar results to layer norm, with reduced compute.
    """

    weight: Float[Array, "dim"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.weight = jnp.ones(shape=(dim,))
        self.eps = eps

    def __call__(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        moment_2 = jnp.mean(x.astype(self.weight.dtype) ** 2, axis=-1, keepdims=True)
        x_normed = x * jax.lax.rsqrt(moment_2 + self.eps)
        return (self.weight * x_normed).astype(x.dtype)


def apply_rotary_embeddings(
    xs: Float[Array, "... seq_len head_dim"], start_idx: int = 0, theta: float = 1e4
):
    # Get the sequence length and head dimension from the input tensor.
    seq_len, head_dim = xs.shape[-2], xs.shape[-1]
    half_dim = head_dim // 2

    inv_freq = theta ** (-jnp.arange(0, half_dim) / half_dim)
    ms = start_idx + jnp.arange(seq_len, dtype=inv_freq.dtype)
    freqs = jnp.outer(ms, inv_freq)

    broadcast_dims = (1,) * (xs.ndim - 2)
    cos_freqs = jnp.cos(freqs).reshape(*broadcast_dims, seq_len, -1)
    sin_freqs = jnp.sin(freqs).reshape(*broadcast_dims, seq_len, -1)

    x1 = xs[..., ::2]  # Even-indexed features
    x2 = xs[..., 1::2]  # Odd-indexed features

    rotated_x1 = x1 * cos_freqs - x2 * sin_freqs
    rotated_x2 = x1 * sin_freqs + x2 * cos_freqs

    xs_rotated = jnp.empty_like(xs)
    xs_rotated = xs_rotated.at[..., ::2].set(rotated_x1)
    xs_rotated = xs_rotated.at[..., 1::2].set(rotated_x2)

    return xs_rotated


def init_weights(
    shape: tuple[int, ...],
    key: PRNGKeyArray,
    dtype: jax.typing.DTypeLike = "float32",
) -> Array:
    fan_in, *rest = shape
    std = jnp.sqrt(2 / fan_in)
    return std * jax.random.truncated_normal(
        key=key, lower=-2, upper=2, shape=(fan_in, *rest), dtype=dtype
    )


class KVCache(NamedTuple):
    # [num_layers, batch, max_seq_len, num_heads, head_dim]
    k: jax.Array
    v: jax.Array

    # Number of tokens currently stored.
    position: jax.Array


def init_kv_cache(
    config: LLaMAConfig,
    batch: int,
    max_seq_len: int,
    dtype=jnp.bfloat16,
) -> KVCache:
    shape = (
        config.num_layers,
        batch,
        max_seq_len,
        config.attention_num_heads,
        config.attention_head_dim,
    )

    return KVCache(
        k=jnp.zeros(shape, dtype=dtype),
        v=jnp.zeros(shape, dtype=dtype),
        position=jnp.array(0, dtype=jnp.int32),
    )


def cache_write(
    cache: KVCache,
    layer_idx: int,
    k: jax.Array,  # [seq_len, heads, head_dim]
    v: jax.Array,
) -> KVCache:
    # Cast once when storing. For cuDNN we want a BF16 cache.
    k = k.astype(cache.k.dtype)
    v = v.astype(cache.v.dtype)

    # Update stays fixed-shape. No concat, no reallocation-by-length.
    new_k = jax.lax.dynamic_update_slice(
        cache.k,
        k[None],
        (layer_idx, 0, cache.position, 0, 0),
    )

    new_v = jax.lax.dynamic_update_slice(
        cache.v,
        v[None],
        (layer_idx, 0, cache.position, 0, 0),
    )

    return cache._replace(
        k=new_k,
        v=new_v,
    )
