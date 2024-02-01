from typing import NamedTuple


class LLaMAConfig(NamedTuple):
    num_layers: int
    size_vocab: int
    size_layer: int
    num_attention_heads: int
    size_attention_heads: int
    size_hidden: int
