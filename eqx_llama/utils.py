from typing import Dict, NamedTuple, Optional, Tuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
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


LayerKVCache: TypeAlias = Tuple[Optional[jax.Array], Optional[jax.Array]]
KVCacheDict: TypeAlias = Dict[str, LayerKVCache]


@jtu.register_pytree_node_class
class KVCache:
    _state: KVCacheDict

    def __init__(self, initial_state: Optional[KVCacheDict] = None):
        self._state = initial_state if initial_state is not None else {}

    def get(self, layer_key: str) -> LayerKVCache:
        """Gets the cache tuple (k, v) for a layer, returning (None, None) if absent."""
        return self._state.get(layer_key, (None, None))

    def set(self, layer_key: str, ks: jax.Array, vs: jax.Array) -> "KVCache":
        """Updates the cache."""
        return KVCache(self._state | {layer_key: (ks, vs)})

    def tree_flatten(self):
        children = list(self._state.values())
        aux_data = list(self._state.keys())
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        keys = aux_data
        state_dict = dict(zip(keys, children))
        return cls(state_dict)

    def __repr__(self):
        count = len(self._state)
        layer_reprs = []
        for layer_key, (ks, vs) in self._state.items():
            ks_shape = ks.shape if ks is not None else None
            layer_reprs.append(f"L{str(layer_key)[-4:]}:(k={ks_shape})")
        layers_str = ", ".join(layer_reprs)
        return f"KVCache(count={count}, layers=[{layers_str}])"


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


def safe_concat(left: Array | None, right: Array) -> Array:
    if left is None:
        return right
    return jnp.concat([left, right], axis=0)
