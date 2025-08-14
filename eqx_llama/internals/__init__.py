from .attention_regular import mha
from .mha_pallas import mha as mha_pallas

__all__ = ["mha", "mha_pallas"]
