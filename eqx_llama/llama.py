from typing import Literal

import equinox as eqx
import jax
from jaxtyping import Array, Float, Integer, PRNGKeyArray

from .llama_attention import AttentionModule
from .llama_feed_forward import FeedForwardModule
from .llama_head import LLaMAHead
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
        attn_implementation: Literal["pallas", "regular"] = "regular",
    ) -> tuple[Float[Array, " seq_len layer_dim"], KVCache | None]:
        attention_out, cache = self.attention_module(
            xs, cache, attn_implementation=attn_implementation
        )
        xs = xs + attention_out
        xs = xs + self.feed_forward_module(xs)
        return xs, cache


class LLaMA(eqx.Module):
    embeddings: eqx.nn.Embedding
    layers: list[LLaMALayer]
    head: LLaMAHead

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: PRNGKeyArray,
        dtype: jax.typing.DTypeLike = "float32",
    ):
        k1, k2, key = jax.random.split(key, 3)
        self.embeddings = eqx.nn.Embedding(
            config.vocab_size, config.layer_dim, key=k1, dtype=dtype
        )
        self.head = LLaMAHead(config, key=k2, dtype=dtype)

        key, *ks = jax.random.split(key, config.num_layers + 1)
        self.layers = [LLaMALayer(config, key=k, dtype=dtype) for k in ks]

    def __call__(
        self,
        tokens: Integer[Array, " seq_len"],
        cache: KVCache | None = None,
        attn_implementation: Literal["pallas", "regular"] = "regular",
    ) -> tuple[Float[Array, " seq_len vocab_size"], KVCache | None]:
        xs = jax.vmap(self.embeddings)(tokens)

        for layer in self.layers:
            xs, cache = layer(xs, cache, attn_implementation=attn_implementation)

        out = jax.vmap(self.head, in_axes=(0))(xs)

        return out, cache
