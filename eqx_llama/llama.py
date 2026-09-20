from typing import Literal

import equinox as eqx
import jax
from jaxtyping import Array, Float, Integer, PRNGKeyArray

from .llama_attention import AttentionModule
from .llama_feed_forward import FeedForwardModule
from .llama_head import LLaMAHead
from .utils import KVCache, LLaMAConfig


class LLaMALayer(eqx.Module):
    attn: AttentionModule
    ffn: FeedForwardModule

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        layer_idx: int,
        key: PRNGKeyArray,
        dtype: jax.typing.DTypeLike = "float32",
    ):
        k1, k2, key = jax.random.split(key, 3)
        self.attn = AttentionModule(config, layer_idx=layer_idx, key=k1, dtype=dtype)
        self.ffn = FeedForwardModule(config, key=k2, dtype=dtype)


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
        self.layers = [
            LLaMALayer(config, layer_idx=i, key=k, dtype=dtype)
            for i, k in enumerate(ks)
        ]

    def embed(self, tokens):
        return jax.vmap(self.embeddings)(tokens)

    def __call__(
        self,
        tokens: Integer[Array, " seq_len"],
        cache: KVCache | None = None,
        attn_implementation: Literal["cudnn", "regular"] = "regular",
    ) -> tuple[Float[Array, " seq_len vocab_size"], KVCache | None]:
        xs = self.embed(tokens)

        for layer in self.layers:
            attn_out, cache = layer.attn(xs, cache, attn_implementation)
            xs = xs + attn_out
            xs = xs + layer.ffn(xs)

        if cache is not None:
            cache = cache._replace(position=cache.position + tokens.shape[0])

        out = self.head(xs)

        return out, cache
