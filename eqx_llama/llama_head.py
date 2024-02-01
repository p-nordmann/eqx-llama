import equinox as eqx
import jax
from beartype import beartype
from jaxtyping import Array, Float32, PRNGKeyArray, jaxtyped

from .llama_config import LLaMAConfig
from .normalization import RMSLayerNorm


class LLaMAHead(eqx.Module):
    norm: RMSLayerNorm
    linear: eqx.nn.Linear

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.norm = RMSLayerNorm(config.size_layer)

        key_linear, key = jax.random.split(key)
        self.linear = eqx.nn.Linear(
            config.size_layer,
            config.size_vocab,
            use_bias=False,
            key=key_linear,
        )

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float32[Array, " size_layer"],
        enable_dropout: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> Float32[Array, " size_vocab"]:
        x_normalized = self.norm(x)
        return self.linear(x_normalized)
