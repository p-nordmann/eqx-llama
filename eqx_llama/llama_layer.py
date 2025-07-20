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
        dtype: jax.typing.DTypeLike = "float32",
    ):
        k1, k2, key = jax.random.split(key, 3)

        self.attention_module = AttentionModule(config, key=k1, dtype=dtype)
        self.feed_forward_module = FeedForwardModule(config, key=k2, dtype=dtype)

    def __call__(
        self,
        xs: Float[Array, " seq_len layer_dim"],
        cache: KVCache | None,
        attn_implementation: Literal["xla", "cudnn", "pallas"] = "xla",
    ) -> tuple[Float[Array, " seq_len layer_dim"], KVCache | None]:
        attention_out, cache = self.attention_module(
            xs, cache, attn_implementation=attn_implementation
        )
        xs = xs + attention_out
        xs = xs + self.feed_forward_module(xs)
        return xs, cache
