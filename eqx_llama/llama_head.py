import chex
import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from .normalization import RMSLayerNorm
from .utils import LLaMAConfig, init_weights


class LLaMAHead(eqx.Module):
    norm: RMSLayerNorm
    weights: Array

    size_layer: int = eqx.field(static=True)
    size_vocab: int = eqx.field(static=True)

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: PRNGKeyArray,
    ):
        k1, key = jax.random.split(key)

        self.norm = RMSLayerNorm(config.size_layer)
        self.weights = init_weights((config.size_layer, config.size_vocab), k1)

        self.size_layer = config.size_layer
        self.size_vocab = config.size_vocab

    def __call__(
        self,
        x: Float[Array, " size_layer"],
    ) -> Float[Array, " size_vocab"]:
        x_normalized = self.norm(x)
        out = x_normalized @ self.weights

        chex.assert_shape([x, x_normalized], (self.size_layer,))
        chex.assert_shape([out], (self.size_vocab,))

        return out
