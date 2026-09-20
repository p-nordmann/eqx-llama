# eqx_llama/llama.py

import equinox as eqx
import jax
import jax.numpy as jnp

from .cache import KVCache, init_kv_cache
from .config import (
    BF16_POLICY,
    AttentionBackend,
    LLaMAConfig,
    PrecisionPolicy,
)
from .llama_attention import AttentionModule
from .llama_feed_forward import FeedForwardModule
from .rope import rope_cos_sin
from .utils import RMSNorm, init_weight, linear


class LLaMALayer(eqx.Module):
    attn: AttentionModule
    ffn: FeedForwardModule

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: jax.Array,
    ):
        k_attn, k_ffn = jax.random.split(key)

        self.attn = AttentionModule(
            config,
            key=k_attn,
        )

        self.ffn = FeedForwardModule(
            config,
            key=k_ffn,
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
        attn = self.attn.forward(
            x,
            cos,
            sin,
            seq_lengths=seq_lengths,
            policy=policy,
            backend=backend,
        )

        x = x + attn.astype(policy.residual)

        ffn = self.ffn(
            x,
            policy,
        )

        x = x + ffn.astype(policy.residual)

        return x

    def prefill(
        self,
        x: jax.Array,
        layer_cache,
        cos: jax.Array,
        sin: jax.Array,
        *,
        seq_lengths: jax.Array,
        policy: PrecisionPolicy,
        backend: AttentionBackend,
    ):
        attn, layer_cache = self.attn.prefill(
            x,
            layer_cache,
            cos,
            sin,
            seq_lengths=seq_lengths,
            policy=policy,
            backend=backend,
        )

        x = x + attn.astype(policy.residual)

        x = x + self.ffn(
            x,
            policy,
        ).astype(policy.residual)

        return x, layer_cache

    def decode(
        self,
        x: jax.Array,
        layer_cache,
        lengths: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        *,
        policy: PrecisionPolicy,
        backend: AttentionBackend,
    ):
        attn, layer_cache = self.attn.decode(
            x,
            layer_cache,
            lengths,
            cos,
            sin,
            policy=policy,
            backend=backend,
        )

        x = x + attn.astype(policy.residual)

        x = x + self.ffn(
            x,
            policy,
        ).astype(policy.residual)

        return x, layer_cache


class LLaMA(eqx.Module):
    # FP32 master embedding.
    token_embedding: jax.Array

    layers: tuple[LLaMALayer, ...]
    final_norm: RMSNorm

    # None means embedding tying.
    lm_head: jax.Array | None

    config: LLaMAConfig = eqx.field(static=True)

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: jax.Array,
    ):
        self.config = config

        keys = jax.random.split(
            key,
            config.num_layers + 2,
        )

        embedding_key = keys[0]
        head_key = keys[1]
        layer_keys = keys[2:]

        self.token_embedding = init_weight(
            embedding_key,
            (
                config.vocab_size,
                config.hidden_dim,
            ),
            config.init_std,
        )

        self.layers = tuple(
            LLaMALayer(
                config,
                key=k,
            )
            for k in layer_keys
        )

        self.final_norm = RMSNorm(
            config.hidden_dim,
            config.rms_norm_eps,
        )

        if config.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = init_weight(
                head_key,
                (
                    config.hidden_dim,
                    config.vocab_size,
                ),
                config.init_std,
            )

    def init_cache(
        self,
        batch_size: int,
        *,
        max_seq_len: int | None = None,
        policy: PrecisionPolicy = BF16_POLICY,
    ) -> KVCache:
        return init_kv_cache(
            self.config,
            batch_size,
            max_seq_len=max_seq_len,
            dtype=policy.cache,
        )

    def _embed(
        self,
        tokens: jax.Array,
        policy: PrecisionPolicy,
    ) -> jax.Array:
        x = self.token_embedding[tokens]
        return x.astype(policy.residual)

    def _logits(
        self,
        x: jax.Array,
        policy: PrecisionPolicy,
    ) -> jax.Array:
        x = self.final_norm(
            x,
            policy.compute,
        )

        if self.lm_head is None:
            weight = self.token_embedding.T
        else:
            weight = self.lm_head

        logits = linear(
            x,
            weight,
            policy.compute,
        )

        return logits.astype(policy.logits)

    def __call__(
        self,
        tokens: jax.Array,  # [B,T]
        *,
        seq_lengths: jax.Array | None = None,
        policy: PrecisionPolicy = BF16_POLICY,
        backend: AttentionBackend = "xla",
    ) -> jax.Array:
        """Training / regular full-sequence forward."""
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")

        b, t = tokens.shape

        if seq_lengths is not None:
            seq_lengths = seq_lengths.astype(jnp.int32)
            seq_lengths = eqx.error_if(
                seq_lengths,
                jnp.any((seq_lengths < 1) | (seq_lengths > t)),
                "invalid sequence lengths",
            )

        positions = jnp.broadcast_to(
            jnp.arange(t, dtype=jnp.int32),
            (b, t),
        )

        cos, sin = rope_cos_sin(
            positions,
            self.config.head_dim,
            self.config.rope_theta,
        )

        x = self._embed(
            tokens,
            policy,
        )

        for layer in self.layers:
            x = layer.forward(
                x,
                cos,
                sin,
                seq_lengths=seq_lengths,
                policy=policy,
                backend=backend,
            )

        return self._logits(
            x,
            policy,
        )

    def prefill(
        self,
        tokens: jax.Array,  # [B,T]
        cache: KVCache,
        *,
        seq_lengths: jax.Array | None = None,
        policy: PrecisionPolicy = BF16_POLICY,
        backend: AttentionBackend = "xla",
    ) -> tuple[jax.Array, KVCache]:
        """Fill an empty cache with a prompt."""
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")

        b, t = tokens.shape

        if cache.lengths.shape != (b,):
            raise ValueError("cache batch size mismatch")

        tokens = eqx.error_if(
            tokens,
            jnp.any(cache.lengths != 0),
            "prefill() requires an empty cache",
        )

        if seq_lengths is None:
            seq_lengths = jnp.full(
                (b,),
                t,
                dtype=jnp.int32,
            )
        else:
            seq_lengths = seq_lengths.astype(jnp.int32)

        seq_lengths = eqx.error_if(
            seq_lengths,
            jnp.any((seq_lengths < 1) | (seq_lengths > t)),
            "invalid prompt lengths",
        )

        positions = jnp.broadcast_to(
            jnp.arange(t, dtype=jnp.int32),
            (b, t),
        )

        cos, sin = rope_cos_sin(
            positions,
            self.config.head_dim,
            self.config.rope_theta,
        )

        x = self._embed(
            tokens,
            policy,
        )

        new_layers = []

        for i, layer in enumerate(self.layers):
            x, layer_cache = layer.prefill(
                x,
                cache.layers[i],
                cos,
                sin,
                seq_lengths=seq_lengths,
                policy=policy,
                backend=backend,
            )

            new_layers.append(layer_cache)

        cache = KVCache(
            layers=tuple(new_layers),
            lengths=seq_lengths,
        )

        return (
            self._logits(x, policy),
            cache,
        )

    def decode(
        self,
        tokens: jax.Array,  # [B] or [B,1]
        cache: KVCache,
        *,
        policy: PrecisionPolicy = BF16_POLICY,
        backend: AttentionBackend = "xla",
    ) -> tuple[jax.Array, KVCache]:
        """
        Decode exactly one token per batch member.

        Returns logits [B,V].
        """
        if tokens.ndim == 1:
            tokens = tokens[:, None]

        if tokens.ndim != 2 or tokens.shape[1] != 1:
            raise ValueError("decode expects [B] or [B,1]")

        b = tokens.shape[0]

        if cache.lengths.shape != (b,):
            raise ValueError("cache batch size mismatch")

        capacity = cache.layers[0].k.shape[1]

        tokens = eqx.error_if(
            tokens,
            jnp.any(cache.lengths >= capacity),
            "KV cache is full",
        )

        # Different members of the batch may be at different
        # absolute positions.
        positions = cache.lengths[:, None]

        cos, sin = rope_cos_sin(
            positions,
            self.config.head_dim,
            self.config.rope_theta,
        )

        x = self._embed(
            tokens,
            policy,
        )

        new_layers = []

        for i, layer in enumerate(self.layers):
            x, layer_cache = layer.decode(
                x,
                cache.layers[i],
                cache.lengths,
                cos,
                sin,
                policy=policy,
                backend=backend,
            )

            new_layers.append(layer_cache)

        cache = KVCache(
            layers=tuple(new_layers),
            lengths=cache.lengths + 1,
        )

        logits = self._logits(
            x,
            policy,
        )

        return logits[:, 0, :], cache
