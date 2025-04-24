from typing import Literal

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from .kv_cache import KVCache
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
        cache: KVCache,
    ) -> tuple[Float[Array, " seq_len size_layer"], KVCache]:
        attention_out, cache = self.attention_module(xs, cache)
        xs = xs + attention_out
        xs = xs + self.feed_forward_module(xs)
        return xs, cache
