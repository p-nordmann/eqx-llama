import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnames=["sm_scale", "causal"])
def mha(qs, ks, vs, sm_scale=1.0, causal: bool = False):
    """Simple implementation of regular MHA."""

    q_len, kv_len = qs.shape[1], ks.shape[1]
    logits = jnp.einsum("bqnh,bknh->bnqk", qs, ks, preferred_element_type=jnp.float32)

    mask = None
    if causal:
        context_len = kv_len - q_len
        q_indices = jnp.arange(q_len) + context_len
        k_indices = jnp.arange(kv_len)
        mask = k_indices <= q_indices[:, None]
        mask = jnp.broadcast_to(mask, logits.shape)

    logits = logits if mask is None else jnp.where(mask, logits, float("-inf"))
    weights = jax.nn.softmax(logits * sm_scale)
    return jnp.einsum(
        "bnqk,bknh->bqnh", weights, vs, preferred_element_type=jnp.float32
    )
