from typing import Optional

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from .llama_config import LLaMAConfig
from .normalization import RMSLayerNorm


class FeedForwardModule(eqx.Module):
    norm: RMSLayerNorm
    linear_in_1: eqx.nn.Linear
    linear_in_2: eqx.nn.Linear
    linear_out: eqx.nn.Linear

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.norm = RMSLayerNorm(config.size_layer)

        key_linear, key = jax.random.split(key)
        self.linear_in_1 = eqx.nn.Linear(
            config.size_layer,
            config.size_hidden,
            use_bias=False,
            key=key_linear,
        )

        key_linear, key = jax.random.split(key)
        self.linear_in_2 = eqx.nn.Linear(
            config.size_layer,
            config.size_hidden,
            use_bias=False,
            key=key_linear,
        )

        key_linear, key = jax.random.split(key)
        self.linear_out = eqx.nn.Linear(
            config.size_hidden,
            config.size_layer,
            use_bias=False,
            key=key_linear,
        )

    def __call__(
        self,
        xs: Float[Array, " seq_len size_layer"],
        enable_dropout: bool = False,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, " seq_len size_layer"]:
        xs_normalized = jax.vmap(self.norm)(xs)
        hidden_1 = jax.vmap(self.linear_in_1)(xs_normalized)
        hidden_2 = jax.vmap(self.linear_in_2)(xs_normalized)
        hidden_after_swiglu = jax.nn.silu(hidden_1) * hidden_2
        return jax.vmap(self.linear_out)(hidden_after_swiglu)
