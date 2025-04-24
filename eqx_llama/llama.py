from typing import Literal

import equinox as eqx
import jax
from jaxtyping import Array, Float, Integer, PRNGKeyArray

from .llama_head import LLaMAHead
from .llama_layer import LLaMALayer
from .utils import KVCache, LLaMAConfig


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
        cache: KVCache,
        attn_implementation: Literal["xla", "cudnn"] = "xla",
    ) -> tuple[Float[Array, " seq_len vocab_size"], KVCache]:
        xs = jax.vmap(self.embeddings)(tokens)

        for layer in self.layers:
            xs, cache = layer(xs, cache, attn_implementation=attn_implementation)

        out = jax.vmap(self.head, in_axes=(0))(xs)

        return out, cache
