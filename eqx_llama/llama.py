from typing import Literal

import equinox as eqx
import jax
from jaxtyping import Array, Float, Integer, PRNGKeyArray

from .llama_config import LLaMAConfig
from .llama_head import LLaMAHead
from .llama_layer import LLaMALayer


class LLaMA(eqx.Module):
    embeddings: eqx.nn.Embedding
    layers: list[LLaMALayer]
    head: LLaMAHead

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: PRNGKeyArray,
        attn_implementation: Literal["xla", "cudnn"] = "xla",
    ):
        key_embeddings, key = jax.random.split(key)
        self.embeddings = eqx.nn.Embedding(
            config.size_vocab,
            config.size_layer,
            key=key_embeddings,
        )

        self.layers = []
        for _ in range(config.num_layers):
            key_layer, key = jax.random.split(key)
            self.layers.append(
                LLaMALayer(
                    config,
                    key=key_layer,
                    attn_implementation=attn_implementation,
                )
            )

        key_head, key = jax.random.split(key)
        self.head = LLaMAHead(
            config,
            key=key_head,
        )

    def __call__(
        self,
        tokens: Integer[Array, " seq_len"],
    ) -> Float[Array, " seq_len size_vocab"]:
        xs = jax.vmap(self.embeddings)(tokens)

        for layer in self.layers:
            xs = layer(xs)

        out = jax.vmap(self.head, in_axes=(0))(xs)

        return out
