import jax
import jax.numpy as jnp

from eqx_llama import KVCache, LLaMA, LLaMAConfig
from eqx_llama.llama_attention import AttentionModule

atol, rtol = 1e-6, 1e-6

mini_config = LLaMAConfig(
    num_layers=2,
    vocab_size=8,
    layer_dim=2,
    attention_num_heads=1,
    attention_head_dim=2,
    feed_forward_dim=4,
)

key = jax.random.PRNGKey(1)
key, k1, k2 = jax.random.split(key, 3)


def test_full_llama_with_cache():
    model = LLaMA(config=mini_config, key=k1)
    tokens = jax.random.randint(k2, (2,), 0, mini_config.vocab_size)

    want, _ = model(tokens, None)

    cache = KVCache()
    out_1, cache = model(tokens[:1], cache)
    out_2, cache = model(tokens[1:2], cache)
    got = jnp.concat([out_1, out_2], axis=0)

    assert jnp.allclose(got, want, rtol=rtol, atol=atol)


def test_attention_module_with_cache():
    attn = AttentionModule(mini_config, key=k1)
    xs = jax.random.normal(k2, (2, mini_config.layer_dim))

    want, _ = attn(xs, None)

    cache = KVCache()
    out_1, cache = attn(xs[:1], cache)
    out_2, cache = attn(xs[1:2], cache)
    got = jnp.concat([out_1, out_2], axis=0)

    assert jnp.allclose(got, want, rtol=rtol, atol=atol)
