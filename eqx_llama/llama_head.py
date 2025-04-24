import chex
import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from .utils import LLaMAConfig, RMSLayerNorm, init_weights


class LLaMAHead(eqx.Module):
    norm: RMSLayerNorm
    weights: Array

    layer_dim: int = eqx.field(static=True)
    vocab_size: int = eqx.field(static=True)

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: PRNGKeyArray,
        dtype: jax.typing.DTypeLike = "float32",
    ):
        k1, key = jax.random.split(key)

        self.norm = RMSLayerNorm(config.layer_dim)
        self.weights = init_weights((config.layer_dim, config.vocab_size), k1, dtype)

        self.layer_dim = config.layer_dim
        self.vocab_size = config.vocab_size

    def __call__(
        self,
        x: Float[Array, " layer_dim"],
    ) -> Float[Array, " vocab_size"]:
        x_normalized = self.norm(x)
        out = x_normalized @ self.weights

        chex.assert_shape([x, x_normalized], (self.layer_dim,))
        chex.assert_shape([out], (self.vocab_size,))

        return out
