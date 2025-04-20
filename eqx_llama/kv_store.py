from typing import Literal, Protocol

import equinox as eqx
import jax
from jaxtyping import Array, Float


class KVStore(Protocol):
    def add(self, key: Array, value: Array): ...
    def query(self, query: Array) -> Array: ...

    def add_many(self, keys: Array, values: Array): ...
    def query_many(self, queries: Array) -> Array: ...

    @staticmethod
    def specialize(store: "KVStore", name: str) -> "KVStore": ...


class SelfAttentionKVStore(eqx.Module): ...


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
