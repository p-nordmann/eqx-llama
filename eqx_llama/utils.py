import equinox as eqx
import jax
import jax.numpy as jnp


def init_weight(
    key: jax.Array,
    shape: tuple[int, ...],
    std: float,
) -> jax.Array:
    """All trainable parameters are stored in FP32."""
    return (
        jax.random.normal(
            key,
            shape=shape,
            dtype=jnp.float32,
        )
        * std
    )


def linear(
    x: jax.Array,
    weight: jax.Array,
    compute_dtype,
) -> jax.Array:
    """
    FP32 master weight -> BF16/FP16 compute weight at use site.

    The result remains in compute_dtype.
    """
    x = x.astype(compute_dtype)
    weight = weight.astype(compute_dtype)
    return jnp.matmul(x, weight)


class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float):
        self.weight = jnp.ones((dim,), dtype=jnp.float32)
        self.eps = eps

    def __call__(
        self,
        x: jax.Array,
        output_dtype,
    ) -> jax.Array:
        # Do normalization statistics and scale application in FP32.
        x32 = x.astype(jnp.float32)

        mean_square = jnp.mean(
            jnp.square(x32),
            axis=-1,
            keepdims=True,
        )

        x32 = x32 * jax.lax.rsqrt(mean_square + self.eps)
        x32 = x32 * self.weight

        return x32.astype(output_dtype)
