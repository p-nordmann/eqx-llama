import jax
import jax.numpy as jnp

from eqx_llama.utils import RMSLayerNorm, apply_rotary_embeddings


def test_rms_layernorm():
    eps = 1e-2
    norm = RMSLayerNorm(10)
    x = jnp.reshape(jnp.arange(0, 20, dtype=jnp.float32), shape=(2, 10))
    got = jax.vmap(norm)(x)
    want = x / jnp.array([[5.339] * 10, [14.782] * 10], dtype=jnp.float32)
    assert jnp.allclose(got, want, atol=eps).item()


def test_apply_rotary_embeddings():
    eps = 1e-2

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
    print(got, want, sep="\n")
    assert jnp.allclose(got, want, atol=eps).item()

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
    print(got, want, sep="\n")
    assert jnp.allclose(got, want, atol=eps).item()
