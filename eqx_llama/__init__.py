from beartype.claw import beartype_this_package

beartype_this_package()


from .llama import LLaMA
from .llama_config import LLaMAConfig

__all__ = ["LLaMA", "LLaMAConfig"]
