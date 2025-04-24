import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from eqx_llama import LLaMA, LLaMAConfig
from eqx_llama.kv_cache import KVCache


def compute_loss(model, cache, inputs):
    outputs, _ = jax.vmap(model, in_axes=(0, None))(inputs, cache)
    return jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(outputs[:, :-1], inputs[:, 1:])
    )


@eqx.filter_jit
def make_step(model, cache, inputs, opt, opt_state):
    grads = eqx.filter_grad(compute_loss)(model, cache, inputs)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state


@eqx.filter_jit
def make_eval_step(model, cache, inputs):
    return compute_loss(model, cache, inputs)


def make_epoch(data, window_size, batch_size, *, key):
    n = data.shape[0]

    # Make permutation of the data.
    key, key_permutation = jax.random.split(key)
    idx = jax.random.permutation(x=data.shape[0], key=key_permutation)

    # Build batches.
    for k in range(0, n - window_size, batch_size):
        if k + batch_size > n - window_size:  # Skip the end
            break
        batch = jnp.zeros((batch_size, window_size), dtype=int)
        for j in range(window_size):
            batch = batch.at[:, j].set(data[idx[k : k + batch_size] + j])
        yield batch


def test_training_sinusoid():
    """Training on a sinusoid to make sure it learns something."""

    # Constants
    seed = 0
    n = 10_000
    n_train = 8_000
    sine_period = 100
    noise_std = 0.3
    config = LLaMAConfig(
        num_layers=2,
        size_vocab=90,
        size_layer=50,
        num_attention_heads=5,
        size_attention_heads=10,
        size_hidden=300,
    )
    learning_rate = 1e-3
    batch_size = 100
    window_size = 40
    expected_improvement_factor = (
        1.5  # Completely arbitrary, but loss should significatively improve
    )

    # Make PRNGKey.
    key = jax.random.PRNGKey(seed)

    # Make sinusoidal data.
    key, key_data = jax.random.split(key)
    data = (
        jnp.sin(jnp.arange(n) * (2 * jnp.pi) / sine_period)
        + jax.random.normal(shape=(n,), key=key_data) * noise_std
    )
    data = (data - jnp.min(data)) / (jnp.max(data) - jnp.min(data))
    data = jnp.digitize(data, jnp.arange(30) / 30) - 1
    data_train, data_test = data[:n_train], data[n_train:]

    # Make model.
    key, key_model = jax.random.split(key)
    model = LLaMA(
        config=config,
        key=key_model,
        attn_implementation="xla",
    )
    cache = KVCache()

    # Make optimizer.
    opt = optax.adam(learning_rate)
    opt_state = opt.init(model)

    # Eval model before training.
    key, key_epoch = jax.random.split(key)
    losses_before = []
    for inputs in make_epoch(
        data=data_test, window_size=window_size, batch_size=batch_size, key=key_epoch
    ):
        loss = make_eval_step(model, cache, inputs)
        losses_before.append(loss)

    # Train model after training.
    key, key_epoch = jax.random.split(key)
    for inputs in make_epoch(
        data=data_train, window_size=window_size, batch_size=batch_size, key=key_epoch
    ):
        model, opt_state = make_step(model, cache, inputs, opt, opt_state)

    # Eval model.
    key, key_epoch = jax.random.split(key)
    losses_after = []
    for inputs in make_epoch(
        data=data_test, window_size=window_size, batch_size=batch_size, key=key_epoch
    ):
        loss = make_eval_step(model, cache, inputs)
        losses_after.append(loss)

    # Check that loss is good.
    before, after = (
        jnp.mean(jnp.stack(losses_before)),
        jnp.mean(jnp.stack(losses_after)),
    )
    assert before / expected_improvement_factor > after
