from typing import Dict, NamedTuple, Optional, Tuple, TypeAlias

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float


class LLaMAConfig(NamedTuple):
    num_layers: int
    size_vocab: int
    size_layer: int
    num_attention_heads: int
    size_attention_heads: int
    size_hidden: int


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


LayerKVCache: TypeAlias = Tuple[Optional[jax.Array], Optional[jax.Array]]
KVCacheDict: TypeAlias = Dict[int, LayerKVCache]


@jtu.register_pytree_node_class
class KVCache:
    _state: KVCacheDict

    def __init__(self, initial_state: Optional[KVCacheDict] = None):
        self._state = initial_state if initial_state is not None else {}

    def get(self, layer_id: int) -> LayerKVCache:
        """Gets the cache tuple (k, v) for a layer, returning (None, None) if absent."""
        return self._state.get(layer_id, (None, None))

    def set(self, layer_id: int, ks: jax.Array, vs: jax.Array) -> "KVCache":
        """Updates the cache."""
        return KVCache(self._state | {layer_id: (ks, vs)})

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
        for layer_id, (ks, vs) in self._state.items():
            ks_shape = ks.shape if ks is not None else None
            layer_reprs.append(f"L{str(layer_id)[-4:]}:(k={ks_shape})")
        layers_str = ", ".join(layer_reprs)
        return f"KVCache(count={count}, layers=[{layers_str}])"
