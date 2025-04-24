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

    layer_dim: int = eqx.field(static=True)
    feed_forward_dim: int = eqx.field(static=True)

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: PRNGKeyArray,
        dtype: jax.typing.DTypeLike = "float32",
    ):
        k1, k2, k3, key = jax.random.split(key, 4)

        self.norm = RMSLayerNorm(config.layer_dim)
        self.weights_in_1 = init_weights(
            (config.layer_dim, config.feed_forward_dim), k1, dtype
        )
        self.weights_in_2 = init_weights(
            (config.layer_dim, config.feed_forward_dim), k2, dtype
        )
        self.weights_out = init_weights(
            (config.feed_forward_dim, config.layer_dim), k3, dtype
        )

        self.layer_dim = config.layer_dim
        self.feed_forward_dim = config.feed_forward_dim

    def __call__(
        self, xs: Float[Array, " seq_len layer_dim"]
    ) -> Float[Array, " seq_len layer_dim"]:
        seq_len = xs.shape[0]

        xs_normalized = jax.vmap(self.norm)(xs)
        hidden_1 = xs_normalized @ self.weights_in_1
        hidden_2 = xs_normalized @ self.weights_in_2
        hidden_after_swiglu = swiglu(hidden_1, hidden_2)
        out = hidden_after_swiglu @ self.weights_out

        chex.assert_shape([xs, xs_normalized, out], (seq_len, self.layer_dim))
        chex.assert_shape(
            [hidden_1, hidden_2, hidden_after_swiglu], (seq_len, self.feed_forward_dim)
        )

        return out


def swiglu(h1: Array, h2: Array) -> Array:
    return jax.nn.silu(h1) * h2
