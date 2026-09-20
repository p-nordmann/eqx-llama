from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

AttentionBackend = Literal["xla", "cudnn"]


@dataclass(frozen=True)
class LLaMAConfig:
    num_layers: int
    vocab_size: int

    hidden_dim: int
    intermediate_dim: int

    num_heads: int
    num_kv_heads: int
    head_dim: int

    max_seq_len: int = 2048

    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    init_std: float = 0.02

    tie_embeddings: bool = False

    def __post_init__(self):
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if self.hidden_dim != self.num_heads * self.head_dim:
            raise ValueError("hidden_dim must equal num_heads * head_dim")

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")


@dataclass(frozen=True)
class PrecisionPolicy:
    # Parameters are deliberately NOT configurable:
    # model parameters remain float32.
    compute_dtype: str
    residual_dtype: str
    logits_dtype: str
    cache_dtype: str

    @property
    def compute(self):
        return jnp.dtype(self.compute_dtype)

    @property
    def residual(self):
        return jnp.dtype(self.residual_dtype)

    @property
    def logits(self):
        return jnp.dtype(self.logits_dtype)

    @property
    def cache(self):
        return jnp.dtype(self.cache_dtype)


FP32_POLICY = PrecisionPolicy(
    compute_dtype="float32",
    residual_dtype="float32",
    logits_dtype="float32",
    cache_dtype="float32",
)

BF16_POLICY = PrecisionPolicy(
    compute_dtype="bfloat16",
    residual_dtype="float32",
    logits_dtype="float32",
    cache_dtype="bfloat16",
)
