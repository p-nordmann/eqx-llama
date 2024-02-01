import equinox as eqx
import jax
from beartype import beartype
from jaxtyping import Array, Float32, PRNGKeyArray, jaxtyped

from .llama_attention import AttentionModule
from .llama_config import LLaMAConfig
from .llama_feed_forward import FeedForwardModule


class LLaMALayer(eqx.Module):
    attention_module: AttentionModule
    feed_forward_module: FeedForwardModule

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: PRNGKeyArray,
    ):
        key_attention, key = jax.random.split(key)
        self.attention_module = AttentionModule(
            config,
            key=key_attention,
        )

        key_feedforward, key = jax.random.split(key)
        self.feed_forward_module = FeedForwardModule(
            config,
            key=key_feedforward,
        )

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        xs: Float32[Array, " seq_len size_layer"],
        enable_dropout: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> Float32[Array, " seq_len size_layer"]:
        xs = xs + self.attention_module(xs)
        xs = xs + self.feed_forward_module(xs)
        return xs
