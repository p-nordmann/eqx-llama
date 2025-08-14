import jax
import jax.numpy as jnp
import pytest

from eqx_llama.llama_attention import compute_self_attention
from eqx_llama.utils import RMSLayerNorm, apply_rotary_embeddings

atol, rtol = 1e-4, 1e-4


def test_rms_layernorm():
    norm = RMSLayerNorm(10)
    x = jnp.reshape(jnp.arange(0, 20, dtype=jnp.float32), shape=(2, 10))
    got = jax.vmap(norm)(x)
    want = x / jnp.array([[5.339] * 10, [14.782] * 10], dtype=jnp.float32)
    assert jnp.allclose(got, want, atol=atol, rtol=rtol)


def test_apply_rotary_embeddings():
    # Two dimensional rotary embeddings are simple.
    got = apply_rotary_embeddings(
        jnp.array(
            [[0, 0], [0, 1], [1, 0], [1, 1]],
            dtype=jnp.float32,
        )
    )
    want = jnp.array(
        [
            [0, 0],
            [-jnp.sin(1), jnp.cos(1)],
            [jnp.cos(2), jnp.sin(2)],
            [jnp.cos(3) - jnp.sin(3), jnp.sin(3) + jnp.cos(3)],
        ],
        dtype=jnp.float32,
    )
    assert jnp.allclose(got, want, atol=atol, rtol=rtol)

    # Four dimensional rotary embeddings are not that hard either.
    got = apply_rotary_embeddings(
        jnp.array(
            [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]],
            dtype=jnp.float32,
        )
    )
    want = jnp.array(
        [
            [0, 0, 0, 1],
            [0, 0, jnp.cos(1e-2), jnp.sin(1e-2)],
            [-jnp.sin(2), jnp.cos(2), 0, 0],
            [jnp.cos(3) - jnp.sin(3), jnp.sin(3) + jnp.cos(3), 0, 0],
            [-jnp.sin(4), jnp.cos(4), jnp.cos(4e-2), jnp.sin(4e-2)],
        ],
        dtype=jnp.float32,
    )
    assert jnp.allclose(got, want, atol=atol, rtol=rtol)


def test_apply_rotary_embeddings_with_start_index():
    key = jax.random.PRNGKey(0)
    xs = jax.random.normal(key, (128, 16))
    ys = apply_rotary_embeddings(xs)

    for k in range(0, 128, 16):
        zs = apply_rotary_embeddings(xs[k:], start_idx=k)
        assert jnp.allclose(zs, ys[k:], atol=atol, rtol=rtol)


# Four combinations: q_len pow2/non, k_len pow2/non
@pytest.fixture(
    params=[
        ((4, 2, 8), (8, 2, 8)),  # q pow2=4, k pow2=8
        ((4, 2, 8), (5, 2, 8)),  # q pow2=4, k non-pow2=5
        ((3, 2, 8), (8, 2, 8)),  # q non=3, k pow2=8
        ((3, 2, 8), (5, 2, 8)),  # q non=3, k non=5
    ]
)
def attn_shapes(request):
    q_shape, kv_shape = request.param
    key = jax.random.PRNGKey(0)
    kq, kk = jax.random.split(key)
    qs = jax.random.normal(kq, q_shape)
    ks = jax.random.normal(kk, kv_shape)
    vs = jax.random.normal(kk, kv_shape)
    return qs, ks, vs


def test_pallas_self_attention(attn_shapes):
    qs, ks, vs = attn_shapes

    out_xla = compute_self_attention(qs, ks, vs, attn_implementation="xla")
    out_pallas = compute_self_attention(
        qs, ks, vs, attn_implementation="pallas", interpret=True
    )

    assert jnp.allclose(out_pallas, out_xla, atol=atol, rtol=rtol)


def test_pallas_self_attention_backward(attn_shapes):
    qs, ks, vs = attn_shapes

    # jit to speed up jacobian; mark string args as static
    f = jax.jit(
        lambda q, k, v, **kw: compute_self_attention(q, k, v, **kw),
        static_argnames=("attn_implementation", "interpret"),
    )
    out_xla = jax.jacobian(f)(qs, ks, vs, attn_implementation="xla")
    out_pallas = jax.jacobian(f)(
        qs, ks, vs, attn_implementation="pallas", interpret=True
    )

    assert jnp.allclose(out_pallas, out_xla, atol=atol, rtol=rtol)
