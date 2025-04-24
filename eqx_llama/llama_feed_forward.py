import chex
import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from .normalization import RMSLayerNorm
from .utils import LLaMAConfig, init_weights


class FeedForwardModule(eqx.Module):
    norm: RMSLayerNorm
    weights_in_1: Array
    weights_in_2: Array
    weights_out: Array

    size_layer: int = eqx.field(static=True)
    size_hidden: int = eqx.field(static=True)

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2, k3, key = jax.random.split(key, 4)

        self.norm = RMSLayerNorm(config.size_layer)
        self.weights_in_1 = init_weights((config.size_layer, config.size_hidden), k1)
        self.weights_in_2 = init_weights((config.size_layer, config.size_hidden), k2)
        self.weights_out = init_weights((config.size_hidden, config.size_layer), k3)

        self.size_layer = config.size_layer
        self.size_hidden = config.size_hidden

    def __call__(
        self, xs: Float[Array, " seq_len size_layer"]
    ) -> Float[Array, " seq_len size_layer"]:
        seq_len = xs.shape[0]

        xs_normalized = jax.vmap(self.norm)(xs)
        hidden_1 = xs_normalized @ self.weights_in_1
        hidden_2 = xs_normalized @ self.weights_in_2
        hidden_after_swiglu = jax.nn.silu(hidden_1) * hidden_2
        out = hidden_after_swiglu @ self.weights_out

        chex.assert_shape([xs, xs_normalized, out], (seq_len, self.size_layer))
        chex.assert_shape(
            [hidden_1, hidden_2, hidden_after_swiglu], (seq_len, self.size_hidden)
        )

        return out
