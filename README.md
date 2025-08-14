# eqx-llama

LLaMA implementation with Jax and Equinox.

## Installation

Right now, there is no release, you must install eqx-llama from source.

```bash
git clone https://github.com/p-nordmann/eqx-llama
cd eqx-llama
pip install .
```

## Usage

```python
import jax

from eqx_llama import KVCache, LLaMA, LLaMAConfig

key_model, key_data = jax.random.split(jax.random.PRNGKey(0))

config = LLaMAConfig(
    num_layers=4,
    vocab_size=256,
    layer_dim=32,
    attention_num_heads=4,
    attention_head_dim=8,
    feed_forward_dim=128,
)
model = LLaMA(config=config, key=key_model, dtype="float32")

inputs = jax.random.randint(key_data, (128,), 0, 256)

cache = KVCache()
logits, cache = model(inputs, cache)
```

For an example of training LLaMA on text, you can have a look at the example notebook ([notebooks/example.ipynb](./notebooks/example.ipynb)).

## Contributing

If you have a question or issues, please do not hesitate to open an issue, I'll try to answer it as soon as possible. :)

## Known issues

At the moment, I noticed the following issues:

- Issue with the pallas attention implementation when using shapes with dimensions that are not powers of 2.

## License

[MIT](https://choosealicense.com/licenses/mit/)
