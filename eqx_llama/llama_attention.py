import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .internals import mha, mha_cudnn
from .utils import (
    KVCache,
    LLaMAConfig,
    RMSLayerNorm,
    apply_rotary_embeddings,
    cache_write,
    init_weights,
)


class _AttentionWeights(eqx.Module):
    wq: Array
    wk: Array
    wv: Array
    wo: Array

    def __init__(self, wq, wk, wv, wo):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo


class AttentionModule(eqx.Module):
    norm: RMSLayerNorm
    weights: _AttentionWeights

    layer_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    layer_idx: int = eqx.field(static=True)

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        layer_idx: int,
        key: PRNGKeyArray,
        dtype: jax.typing.DTypeLike = "float32",
    ):
        assert (
            config.attention_num_heads * config.attention_head_dim == config.layer_dim
        )
        k1, k2, k3, k4, key = jax.random.split(key, 5)

        self.layer_dim = config.layer_dim
        self.num_heads = config.attention_num_heads
        self.head_dim = config.attention_head_dim

        self.norm = RMSLayerNorm(config.layer_dim)
        self.weights = _AttentionWeights(
            init_weights((config.layer_dim, self.num_heads, self.head_dim), k1, dtype),
            init_weights((config.layer_dim, self.num_heads, self.head_dim), k2, dtype),
            init_weights((config.layer_dim, self.num_heads, self.head_dim), k3, dtype),
            init_weights((config.layer_dim, self.num_heads, self.head_dim), k4, dtype),
        )

        self.layer_idx = layer_idx

    def _compute_embeddings(
        self,
        xs: Float[Array, "seq_len layer_dim"],
        start_index: int = 0,
    ) -> tuple[
        Float[Array, "seq_len num_heads head_dim"],
        Float[Array, "seq_len num_heads head_dim"],
        Float[Array, "seq_len num_heads head_dim"],
    ]:
        apply_rope = jax.vmap(
            lambda h: apply_rotary_embeddings(h, start_index), in_axes=1, out_axes=1
        )
        qs = apply_rope(jnp.einsum("sd,dnh->snh", xs, self.weights.wq))
        ks = apply_rope(jnp.einsum("sd,dnh->snh", xs, self.weights.wk))
        vs = jnp.einsum("sd,dnh->snh", xs, self.weights.wv)
        return qs, ks, vs

    def __call__(
        self,
        xs: Float[Array, "seq_len layer_dim"],
        cache: KVCache | None,
        attn_implementation: Literal["cudnn", "regular"] = "regular",
    ) -> Float[Array, "seq_len layer_dim"]:
        seq_len = xs.shape[0]

        # -----------------------------
        # Normal training path
        # -----------------------------
        if cache is None:
            qs, ks, vs = self._compute_embeddings(
                self.norm(xs),
                start_index=0,
            )
            attn_out = compute_self_attention_padded(
                qs,
                ks,
                vs,
                attn_implementation,
            )

        # -----------------------------
        # Cached inference
        # -----------------------------
        else:
            start = cache.position

            qs, new_ks, new_vs = self._compute_embeddings(
                self.norm(xs),
                start_index=start,
            )

            cache = cache_write(
                cache,
                self.layer_idx,
                new_ks,
                new_vs,
            )

            if seq_len == 1:
                # Decode.
                #
                # Physical K/V length stays max_seq_len.
                # Logical length tells cuDNN which part is real.
                kv_len = cache.position + 1

                attn_out = cached_decode_attention(
                    qs,
                    cache.k[self.layer_idx],
                    cache.v[self.layer_idx],
                    kv_len,
                )

            else:
                # Initial prompt prefill.
                #
                # Don't make attention look at the fixed-size cache at all.
                # We already have exactly the K/V we need here.
                #
                # Deliberately only supporting one-shot prefill for now.
                attn_out = compute_self_attention_padded(
                    qs,
                    new_ks,
                    new_vs,
                    attn_implementation,
                )

        out = jnp.einsum("snh,dnh->sd", attn_out, self.weights.wo)

        return out, cache


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def compute_self_attention_padded(
    qs: Float[Array, "seq_len num_heads head_dim"],
    ks: Float[Array, "kv_len num_heads head_dim"],
    vs: Float[Array, "kv_len num_heads head_dim"],
    attn_implementation: Literal["cudnn", "regular"] = "regular",
    **kwargs,
) -> Float[Array, "seq_len num_heads head_dim"]:
    return compute_self_attention(
        qs=qs,
        ks=ks,
        vs=vs,
        attn_implementation=attn_implementation,
        **kwargs,
    )


def compute_self_attention(
    qs: Float[Array, "seq_len num_heads head_dim"],
    ks: Float[Array, "context_len+seq_len num_heads head_dim"],
    vs: Float[Array, "context_len+seq_len num_heads head_dim"],
    attn_implementation: Literal["cudnn", "regular"] = "regular",
    **kwargs,
) -> Float[Array, "seq_len num_heads head_dim"]:
    assert ks.shape[0] >= qs.shape[0], (
        "kv sequence must be at least as long as q sequence"
    )

    if attn_implementation == "cudnn":
        if qs.shape[0] != ks.shape[0]:
            raise ValueError(
                "cudnn attention is currently enabled only for "
                "full-sequence attention; use regular for cached decoding"
            )
        return mha_cudnn(
            qs[None, ...],
            ks[None, ...],
            vs[None, ...],
            sm_scale=1 / math.sqrt(qs.shape[2]),
            causal=True,
        )[0]

    if attn_implementation == "regular":
        return mha(
            qs[None, ...],
            ks[None, ...],
            vs[None, ...],
            sm_scale=1 / math.sqrt(qs.shape[2]),
            causal=True,
        )[0, ...]

    raise ValueError(f"Unexpected attention implementation '{attn_implementation}'")


def cached_decode_attention(
    qs,  # [1, heads, dim]
    ks,  # [capacity, heads, dim]
    vs,
    kv_len,  # scalar
):
    output_dtype = qs.dtype

    # cuDNN SDPA wants FP16/BF16.
    qs = qs.astype(jnp.bfloat16)
    ks = ks.astype(jnp.bfloat16)
    vs = vs.astype(jnp.bfloat16)

    out = jax.nn.dot_product_attention(
        qs[None],
        ks[None],
        vs[None],
        # q_len == 1, so all valid keys are causal.
        is_causal=False,
        query_seq_lengths=jnp.ones(
            (1,),
            dtype=jnp.int32,
        ),
        key_value_seq_lengths=jnp.reshape(
            kv_len,
            (1,),
        ).astype(jnp.int32),
        implementation="cudnn",
    )

    return out[0].astype(output_dtype)
