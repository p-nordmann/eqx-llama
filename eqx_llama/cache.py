import equinox as eqx
import jax
import jax.numpy as jnp

from .config import LLaMAConfig


class LayerKVCache(eqx.Module):
    # B S K H
    k: jax.Array
    v: jax.Array


class KVCache(eqx.Module):
    layers: tuple[LayerKVCache, ...]

    # Actual number of valid tokens for each batch member.
    # Shape: [B]
    lengths: jax.Array


def init_kv_cache(
    config: LLaMAConfig,
    batch_size: int,
    *,
    max_seq_len: int | None = None,
    dtype=jnp.bfloat16,
) -> KVCache:
    max_seq_len = config.max_seq_len if max_seq_len is None else max_seq_len

    shape = (
        batch_size,
        max_seq_len,
        config.num_kv_heads,
        config.head_dim,
    )

    layers = tuple(
        LayerKVCache(
            k=jnp.zeros(shape, dtype=dtype),
            v=jnp.zeros(shape, dtype=dtype),
        )
        for _ in range(config.num_layers)
    )

    lengths = jnp.zeros(
        (batch_size,),
        dtype=jnp.int32,
    )

    return KVCache(
        layers=layers,
        lengths=lengths,
    )


def write_layer_cache(
    cache: LayerKVCache,
    new_k: jax.Array,  # [B, T, K, H]
    new_v: jax.Array,  # [B, T, K, H]
    start: jax.Array,  # [B]
) -> LayerKVCache:
    """
    Writes each batch element at its own logical position.

    All returned arrays have exactly the same shape as the inputs.
    """
    num_new_tokens = new_k.shape[1]
    capacity = cache.k.shape[1]

    start = eqx.error_if(
        start,
        jnp.any(start + num_new_tokens > capacity),
        "KV cache capacity exceeded",
    )

    def write_one(
        dst: jax.Array,  # [S, K, H]
        src: jax.Array,  # [T, K, H]
        pos: jax.Array,  # scalar
    ) -> jax.Array:
        return jax.lax.dynamic_update_slice_in_dim(
            dst,
            src,
            pos,
            axis=0,
        )

    k = jax.vmap(write_one)(
        cache.k,
        new_k,
        start,
    )

    v = jax.vmap(write_one)(
        cache.v,
        new_v,
        start,
    )

    return LayerKVCache(k=k, v=v)
