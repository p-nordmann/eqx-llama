import equinox as eqx
import jax
from beartype import beartype
from jaxtyping import Array, Float32, Integer, PRNGKeyArray, jaxtyped

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
                )
            )

        key_head, key = jax.random.split(key)
        self.head = LLaMAHead(
            config,
            key=key_head,
        )

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        tokens: Integer[Array, " seq_len"],
        enable_dropout: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> Float32[Array, " seq_len size_vocab"]:
        xs = jax.vmap(self.embeddings)(tokens)

        for layer in self.layers:
            key_layer, key = jax.random.split(key) if key else None, None
            xs = layer(xs, enable_dropout, key_layer)

        key_head, key = jax.random.split(key) if key else None, None
        out = jax.vmap(
            self.head,
            in_axes=(0, None, None),
        )(xs, enable_dropout, key_head)

        return out
