import jax
import jax.numpy as jnp


def rope_cos_sin(
    position_ids: jax.Array,  # [B, T]
    head_dim: int,
    theta: float,
) -> tuple[jax.Array, jax.Array]:
    """
    Returns FP32 cos/sin:
        [B, T, head_dim // 2]
    """
    if head_dim % 2:
        raise ValueError("RoPE requires an even head_dim")

    freq_idx = jnp.arange(
        0,
        head_dim,
        2,
        dtype=jnp.float32,
    )

    inv_freq = theta ** (-freq_idx / head_dim)

    angles = position_ids.astype(jnp.float32)[..., None] * inv_freq

    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(
    x: jax.Array,  # [B, T, N, H]
    cos: jax.Array,  # [B, T, H/2]
    sin: jax.Array,  # [B, T, H/2]
) -> jax.Array:
    input_dtype = x.dtype

    x = x.astype(jnp.float32)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Broadcast over heads.
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]

    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos

    # [..., H/2, 2] -> [..., H]
    y = jnp.stack((y_even, y_odd), axis=-1)
    y = y.reshape(x.shape)

    return y.astype(input_dtype)
