# eqx_llama/llama_attention.py

import equinox as eqx
import jax
import jax.numpy as jnp

from .cache import LayerKVCache, write_layer_cache
from .config import (
    AttentionBackend,
    LLaMAConfig,
    PrecisionPolicy,
)
from .rope import apply_rope
from .utils import RMSNorm, init_weight, linear


class AttentionModule(eqx.Module):
    norm: RMSNorm

    # FP32 master parameters.
    w_qkv: jax.Array
    w_out: jax.Array

    num_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: jax.Array,
    ):
        k_qkv, k_out = jax.random.split(key)

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim

        qkv_dim = (config.num_heads + 2 * config.num_kv_heads) * config.head_dim

        self.norm = RMSNorm(
            config.hidden_dim,
            config.rms_norm_eps,
        )

        self.w_qkv = init_weight(
            k_qkv,
            (config.hidden_dim, qkv_dim),
            config.init_std,
        )

        self.w_out = init_weight(
            k_out,
            (
                config.num_heads * config.head_dim,
                config.hidden_dim,
            ),
            config.init_std,
        )

    def _project_qkv(
        self,
        x: jax.Array,  # [B,T,D]
        cos: jax.Array,
        sin: jax.Array,
        policy: PrecisionPolicy,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        x = self.norm(x, policy.compute)

        qkv = linear(
            x,
            self.w_qkv,
            policy.compute,
        )

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        q, k, v = jnp.split(
            qkv,
            (q_size, q_size + kv_size),
            axis=-1,
        )

        b, t, _ = q.shape

        q = q.reshape(
            b,
            t,
            self.num_heads,
            self.head_dim,
        )

        k = k.reshape(
            b,
            t,
            self.num_kv_heads,
            self.head_dim,
        )

        v = v.reshape(
            b,
            t,
            self.num_kv_heads,
            self.head_dim,
        )

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        return q, k, v

    def _project_output(
        self,
        x: jax.Array,  # [B,T,N,H]
        policy: PrecisionPolicy,
    ) -> jax.Array:
        b, t, _, _ = x.shape

        x = x.reshape(
            b,
            t,
            self.num_heads * self.head_dim,
        )

        return linear(
            x,
            self.w_out,
            policy.compute,
        )

    def forward(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        *,
        seq_lengths: jax.Array | None,
        policy: PrecisionPolicy,
        backend: AttentionBackend,
    ) -> jax.Array:
        """
        Training / ordinary full-sequence forward.
        """
        q, k, v = self._project_qkv(
            x,
            cos,
            sin,
            policy,
        )

        if backend == "cudnn":
            q = q.astype(jnp.bfloat16)
            k = k.astype(jnp.bfloat16)
            v = v.astype(jnp.bfloat16)

        attn = jax.nn.dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            query_seq_lengths=seq_lengths,
            key_value_seq_lengths=seq_lengths,
            implementation=backend,
        )

        return self._project_output(
            attn,
            policy,
        )

    def prefill(
        self,
        x: jax.Array,
        layer_cache: LayerKVCache,
        cos: jax.Array,
        sin: jax.Array,
        *,
        seq_lengths: jax.Array,
        policy: PrecisionPolicy,
        backend: AttentionBackend,
    ) -> tuple[jax.Array, LayerKVCache]:
        """
        Prefill an EMPTY cache with a possibly padded prompt.
        """
        q, k, v = self._project_qkv(
            x,
            cos,
            sin,
            policy,
        )

        if backend == "cudnn":
            q = q.astype(jnp.bfloat16)
            k = k.astype(jnp.bfloat16)
            v = v.astype(jnp.bfloat16)

        attn = jax.nn.dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            query_seq_lengths=seq_lengths,
            key_value_seq_lengths=seq_lengths,
            implementation=backend,
        )

        zeros = jnp.zeros_like(seq_lengths)

        layer_cache = write_layer_cache(
            layer_cache,
            k,
            v,
            zeros,
        )

        return (
            self._project_output(attn, policy),
            layer_cache,
        )

    def decode(
        self,
        x: jax.Array,  # [B,1,D]
        layer_cache: LayerKVCache,
        lengths: jax.Array,  # [B]
        cos: jax.Array,
        sin: jax.Array,
        *,
        policy: PrecisionPolicy,
        backend: AttentionBackend,
    ) -> tuple[jax.Array, LayerKVCache]:
        """
        Exactly one new token per batch element.
        """
        if x.shape[1] != 1:
            raise ValueError(
                "decode() supports exactly one token; use prefill() for prompts"
            )

        q, new_k, new_v = self._project_qkv(
            x,
            cos,
            sin,
            policy,
        )

        layer_cache = write_layer_cache(
            layer_cache,
            new_k,
            new_v,
            lengths,
        )

        kv_lengths = lengths + 1
        q_lengths = jnp.ones_like(lengths)

        k = layer_cache.k
        v = layer_cache.v

        if backend == "cudnn":
            q = q.astype(jnp.bfloat16)
            k = k.astype(jnp.bfloat16)
            v = v.astype(jnp.bfloat16)

        # Important:
        #
        # There is NO causal mask here.
        #
        # For one-token decoding every VALID key is in the
        # query's past or is the current token itself.
        #
        # key_value_seq_lengths excludes the unused tail of
        # the fixed-size cache.
        attn = jax.nn.dot_product_attention(
            q,
            layer_cache.k,
            layer_cache.v,
            is_causal=False,
            query_seq_lengths=q_lengths,
            key_value_seq_lengths=kv_lengths,
            implementation=backend,
        )

        return (
            self._project_output(attn, policy),
            layer_cache,
        )
