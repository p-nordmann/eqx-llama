# eqx_llama/llama_feed_forward.py

import equinox as eqx
import jax
import jax.numpy as jnp

from .config import LLaMAConfig, PrecisionPolicy
from .utils import RMSNorm, init_weight, linear


class FeedForwardModule(eqx.Module):
    norm: RMSNorm

    # FP32 masters.
    w_gate_up: jax.Array
    w_down: jax.Array

    intermediate_dim: int = eqx.field(static=True)

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: jax.Array,
    ):
        k1, k2 = jax.random.split(key)

        self.intermediate_dim = config.intermediate_dim

        self.norm = RMSNorm(
            config.hidden_dim,
            config.rms_norm_eps,
        )

        self.w_gate_up = init_weight(
            k1,
            (
                config.hidden_dim,
                2 * config.intermediate_dim,
            ),
            config.init_std,
        )

        self.w_down = init_weight(
            k2,
            (
                config.intermediate_dim,
                config.hidden_dim,
            ),
            config.init_std,
        )

    def __call__(
        self,
        x: jax.Array,
        policy: PrecisionPolicy,
    ) -> jax.Array:
        x = self.norm(
            x,
            policy.compute,
        )

        gate_up = linear(
            x,
            self.w_gate_up,
            policy.compute,
        )

        gate, up = jnp.split(
            gate_up,
            2,
            axis=-1,
        )

        hidden = jax.nn.silu(gate) * up

        return linear(
            hidden,
            self.w_down,
            policy.compute,
        )
