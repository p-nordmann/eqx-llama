from typing import Literal, Optional

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

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
        attn_implementation: Literal["xla", "cudnn"] = "xla",
    ):
        key_attention, key = jax.random.split(key)
        self.attention_module = AttentionModule(
            config,
            key=key_attention,
            attn_implementation=attn_implementation,
        )

        key_feedforward, key = jax.random.split(key)
        self.feed_forward_module = FeedForwardModule(
            config,
            key=key_feedforward,
        )

    def __call__(
        self,
        xs: Float[Array, " seq_len size_layer"],
        enable_dropout: bool = False,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, " seq_len size_layer"]:
        xs = xs + self.attention_module(xs)
        xs = xs + self.feed_forward_module(xs)
        return xs
