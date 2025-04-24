from typing import Dict, Optional, Tuple, TypeAlias

import jax
import jax.tree_util as jtu

LayerKVCache: TypeAlias = Tuple[Optional[jax.Array], Optional[jax.Array]]
KVCacheDict: TypeAlias = Dict[int, LayerKVCache]


@jtu.register_pytree_node_class
class KVCache:
    """
    Manages KV caches using simple concatenation.

    - No pre-allocation or resizing.
    - Cache grows dynamically via jnp.concatenate.
    - Expect JIT recompilation during inference when cache shapes change.
    """

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
        return f"SimpleCacheState(count={count}, layers=[{layers_str}])"
