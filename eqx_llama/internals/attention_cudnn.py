import jax
import jax.numpy as jnp


def mha_cudnn(qs, ks, vs, causal=False):
    """
    Inputs:
        qs: [B, Q, N, H]
        ks: [B, K, N, H]
        vs: [B, K, N, H]

    Output:
        [B, Q, N, H]
    """
    input_dtype = qs.dtype

    qs = qs.astype(jnp.bfloat16)
    ks = ks.astype(jnp.bfloat16)
    vs = vs.astype(jnp.bfloat16)

    out = jax.nn.dot_product_attention(
        qs,
        ks,
        vs,
        is_causal=causal,
        implementation="cudnn",
    )

    return out.astype(input_dtype)
