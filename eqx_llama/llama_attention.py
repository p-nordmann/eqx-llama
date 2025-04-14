from typing import Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .llama_config import LLaMAConfig
from .normalization import RMSLayerNorm


class AttentionModule(eqx.Module):
    norm: RMSLayerNorm
    rope_embeddings: eqx.nn.RotaryPositionalEmbedding
    linear_q: eqx.nn.Linear
    linear_k: eqx.nn.Linear
    linear_v: eqx.nn.Linear
    linear_o: eqx.nn.Linear

    num_attention_heads: int = eqx.field(static=True)
    size_attention_heads: int = eqx.field(static=True)
    attn_implementation: Literal["xla", "cudnn"] = eqx.field(static=True)

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: PRNGKeyArray,
        attn_implementation: Literal["xla", "cudnn"] = "xla",
    ):
        assert (
            config.num_attention_heads * config.size_attention_heads
            == config.size_layer
        )
        self.num_attention_heads = config.num_attention_heads
        self.size_attention_heads = config.size_attention_heads
        self.attn_implementation = attn_implementation

        self.norm = RMSLayerNorm(config.size_layer)

        self.rope_embeddings = eqx.nn.RotaryPositionalEmbedding(
            config.size_attention_heads, 10_000
        )

        key_linear, key = jax.random.split(key)
        self.linear_q = eqx.nn.Linear(
            config.size_layer,
            config.size_layer,
            use_bias=False,
            key=key_linear,
        )

        key_linear, key = jax.random.split(key)
        self.linear_k = eqx.nn.Linear(
            config.size_layer,
            config.size_layer,
            use_bias=False,
            key=key_linear,
        )

        key_linear, key = jax.random.split(key)
        self.linear_v = eqx.nn.Linear(
            config.size_layer,
            config.size_layer,
            use_bias=False,
            key=key_linear,
        )

        key_linear, key = jax.random.split(key)
        self.linear_o = eqx.nn.Linear(
            config.size_layer,
            config.size_layer,
            use_bias=False,
            key=key_linear,
        )

    def _compute_embeddings(
        self,
        xs: Float[Array, " seq_len size_layer"],
        linear: eqx.nn.Linear,
        use_position_embeddings: bool = False,
    ) -> Float[Array, " seq_len num_heads size_heads"]:
        projected_xs = jax.vmap(linear)(xs)
        hs = jnp.reshape(
            projected_xs,
            shape=(-1, self.num_attention_heads, self.size_attention_heads),
        )
        if not use_position_embeddings:
            return hs
        return jax.vmap(self.rope_embeddings, in_axes=1, out_axes=1)(hs)

    def __call__(
        self,
        xs: Float[Array, " seq_len size_layer"],
        enable_dropout: bool = False,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, " seq_len size_layer"]:
        xs_normalized = jax.vmap(self.norm)(xs)
        qs = self._compute_embeddings(
            xs_normalized,
            self.linear_q,
            use_position_embeddings=True,
        )
        ks = self._compute_embeddings(
            xs_normalized,
            self.linear_k,
            use_position_embeddings=True,
        )
        vs = self._compute_embeddings(xs_normalized, self.linear_v)
        attention_out = compute_self_attention(
            qs, ks, vs, attn_implementation=self.attn_implementation
        )
        return jax.vmap(self.linear_o)(jax.lax.collapse(attention_out, 1, 3))


def compute_self_attention(
    qs: Float[Array, " seqlen num_heads head_dim"],
    ks: Float[Array, " seqlen num_heads head_dim"],
    vs: Float[Array, " seqlen num_heads head_dim"],
    *,
    attn_implementation: Literal["xla", "cudnn"] = "xla",
) -> Float[Array, " seqlen num_heads head_dim"]:
    return jax.nn.dot_product_attention(
        qs, ks, vs, is_causal=True, implementation=attn_implementation
    )
