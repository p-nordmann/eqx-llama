import math

import jax
import jax.numpy as jnp
import pytest

from eqx_llama.llama_attention import compute_self_attention

atol, rtol = 1e-6, 1e-6


@pytest.fixture(
    params=[
        ((4, 2, 8), (8, 2, 8)),  # q pow2=4, k pow2=8
        ((4, 2, 8), (5, 2, 8)),  # q pow2=4, k non-pow2=5
        ((3, 2, 8), (8, 2, 8)),  # q non=3, k pow2=8
        ((3, 2, 8), (5, 2, 8)),  # q non=3, k non=5
    ]
)
def attn_inputs(request):
    q_shape, kv_shape = request.param
    key = jax.random.PRNGKey(0)
    kq, kk = jax.random.split(key)
    qs = jax.random.normal(kq, q_shape)
    ks = jax.random.normal(kk, kv_shape)
    vs = jax.random.normal(kk, kv_shape)
    return qs, ks, vs


def reference_attention(qs, ks, vs):
    """Reference self-attention implementation using XLA implementation."""

    # The causal flag in attention implementations does not
    # take into account the case where S and T are not the same.
    # We need to use a mask to take it into account.
    q_len = qs.shape[0]
    kv_len = ks.shape[0]
    context_len = kv_len - q_len
    q_indices = jnp.arange(q_len) + context_len
    k_indices = jnp.arange(kv_len)

    # We only want the queries to attend in the past.
    causal_mask = k_indices <= q_indices[:, None]

    return jax.nn.dot_product_attention(
        qs,
        ks,
        vs,
        is_causal=False,
        mask=causal_mask,
        implementation="xla",
        scale=1 / math.sqrt(qs.shape[2]),
    )


@pytest.mark.cpu
@pytest.mark.gpu
@pytest.mark.tpu
def test_regular_self_attention(attn_inputs):
    qs, ks, vs = attn_inputs

    want = reference_attention(qs, ks, vs)
    got = compute_self_attention(qs, ks, vs, attn_implementation="regular")

    assert jnp.allclose(got, want, atol=atol, rtol=rtol)


@pytest.mark.cpu
def test_pallas_self_attention_cpu(attn_inputs):
    qs, ks, vs = attn_inputs

    want = reference_attention(qs, ks, vs)
    got = compute_self_attention(
        qs, ks, vs, attn_implementation="pallas", interpret=True
    )

    assert jnp.allclose(got, want, atol=atol, rtol=rtol)


@pytest.mark.cpu
def test_pallas_self_attention_backward_cpu(attn_inputs):
    qs, ks, vs = attn_inputs

    def pallas_implementation(qs, ks, vs):
        return compute_self_attention(
            qs, ks, vs, attn_implementation="pallas", interpret=True
        )

    want = jax.jit(jax.jacobian(reference_attention))(qs, ks, vs)
    got = jax.jit(jax.jacobian(pallas_implementation))(qs, ks, vs)

    assert jnp.allclose(got, want, atol=atol, rtol=rtol)


@pytest.mark.gpu
def test_pallas_self_attention_gpu(attn_inputs):
    qs, ks, vs = attn_inputs

    want = reference_attention(qs, ks, vs)
    got = compute_self_attention(qs, ks, vs, attn_implementation="pallas")

    # atol, rtol = 1e-2, 1e-2
    assert jnp.allclose(got, want, atol=atol, rtol=rtol)


@pytest.mark.gpu
def test_pallas_self_attention_backward_gpu(attn_inputs):
    qs, ks, vs = attn_inputs

    def pallas_implementation(qs, ks, vs):
        return compute_self_attention(qs, ks, vs, attn_implementation="pallas")

    want = jax.jit(jax.jacobian(reference_attention))(qs, ks, vs)
    got = jax.jit(jax.jacobian(pallas_implementation))(qs, ks, vs)

    # atol, rtol = 1e-2, 1e-2
    assert jnp.allclose(got, want, atol=atol, rtol=rtol)
