from typing import Literal

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from .llama_attention import AttentionModule
from .llama_feed_forward import FeedForwardModule
from .utils import KVCache, LLaMAConfig


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
        k1, k2, key = jax.random.split(key, 3)

        self.attention_module = AttentionModule(
            config, key=k1, attn_implementation=attn_implementation
        )
        self.feed_forward_module = FeedForwardModule(config, key=k2)

    def __call__(
        self,
        xs: Float[Array, " seq_len size_layer"],
        cache: KVCache,
    ) -> tuple[Float[Array, " seq_len size_layer"], KVCache]:
        attention_out, cache = self.attention_module(xs, cache)
        xs = xs + attention_out
        xs = xs + self.feed_forward_module(xs)
        return xs, cache
