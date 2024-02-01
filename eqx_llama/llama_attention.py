import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float32, PRNGKeyArray, jaxtyped

from .llama_config import LLaMAConfig
from .normalization import RMSLayerNorm


# def rotary_kernel(x: Float32[Array, " 2"], m_theta: float) -> Float32[Array, " 2"]:
def rotary_kernel(x, m_theta):
    """The core operation of rotary embeddings acts over two dimensions."""
    theta_kernel = jnp.array(
        [
            [jnp.cos(m_theta), -jnp.sin(m_theta)],
            [jnp.sin(m_theta), jnp.cos(m_theta)],
        ]
    )
    return theta_kernel @ x


# def generalized_rotary_kernel(
#     x: Float32[Array, " size"], m: Float32, thetas: Float32[Array, " half_size"]
# ) -> Float32[Array, " half_size"]:
def generalized_rotary_kernel(x, m, thetas):
    """Applies the rotary kernel along a vector of even dimension.

    Thetas must be provided.
    """
    pairs_of_xi = jnp.reshape(
        x,
        newshape=(-1, 2),
        order="C",  # Order is critical to be consistent with the original LLaMAs!
    )
    pairs_of_embeddings = jax.vmap(rotary_kernel)(pairs_of_xi, m * thetas)
    return jax.lax.collapse(pairs_of_embeddings, 0, 2)


def apply_rotary_embeddings(
    xs: Float32[Array, " seq_len size"],
    *,
    theta_base: float = 1e4,
    dtype=jnp.float16,
) -> Float32[Array, " seq_len size"]:
    """Applies the rotary kernel through a full sequence with even dimension."""
    half_dim = xs.shape[1] // 2
    ms = jnp.arange(0, xs.shape[0])
    thetas = theta_base ** (-jnp.arange(0, half_dim, dtype=dtype) / half_dim)
    return jax.vmap(
        generalized_rotary_kernel,
        in_axes=[0, 0, None],
    )(xs, ms, thetas)


def compute_attention_scores(
    q: Float32[Array, " head_dim"],
    ks: Float32[Array, " seqlen head_dim"],
    mask: Float32[Array, " seqlen"],
) -> Float32[Array, " seqlen"]:
    head_dim = q.shape[0]
    unnormalized_scores = jnp.inner(q, ks) / jnp.sqrt(head_dim) + mask
    return jax.nn.softmax(unnormalized_scores)


def compute_self_attention(
    qs: Float32[Array, " seqlen head_dim"],
    ks: Float32[Array, " seqlen head_dim"],
    vs: Float32[Array, " seqlen head_dim"],
) -> Float32[Array, " seqlen head_dim"]:
    """Computes the full self-attention.

    Uses vmap rather than einsums or messy transpositions / reshapes...
    """
    mask = jnp.triu(
        jnp.full((qs.shape[0], qs.shape[0]), float("-inf")), k=1
    )  # Uses -inf for numerical stability.
    scores = jax.vmap(compute_attention_scores, in_axes=(0, None, 0))(qs, ks, mask)
    return scores @ vs


class AttentionModule(eqx.Module):
    norm: RMSLayerNorm
    linear_q: eqx.nn.Linear
    linear_k: eqx.nn.Linear
    linear_v: eqx.nn.Linear
    linear_o: eqx.nn.Linear

    num_attention_heads: int = eqx.field(static=True)
    size_attention_heads: int = eqx.field(static=True)

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: PRNGKeyArray,
    ):
        assert (
            config.num_attention_heads * config.size_attention_heads
            == config.size_layer
        )
        self.num_attention_heads = config.num_attention_heads
        self.size_attention_heads = config.size_attention_heads

        self.norm = RMSLayerNorm(config.size_layer)

        key_linear, key = jax.random.split(key)
        self.linear_q = eqx.nn.Linear(
            config.size_layer,
            config.size_layer,
            use_bias=False,
            key=key_linear,
        )

        key_linear, key = jax.random.split(key)
        self.linear_k = eqx.nn.Linear(
            config.size_layer,
            config.size_layer,
            use_bias=False,
            key=key_linear,
        )

        key_linear, key = jax.random.split(key)
        self.linear_v = eqx.nn.Linear(
            config.size_layer,
            config.size_layer,
            use_bias=False,
            key=key_linear,
        )

        key_linear, key = jax.random.split(key)
        self.linear_o = eqx.nn.Linear(
            config.size_layer,
            config.size_layer,
            use_bias=False,
            key=key_linear,
        )

    def _compute_embeddings(
        self,
        xs: Float32[Array, " seq_len size_layer"],
        linear: eqx.nn.Linear,
        use_position_embeddings: bool = False,
    ) -> Float32[Array, " seq_len num_heads size_heads"]:
        projected_xs = jax.vmap(linear)(xs)
        hs = jnp.reshape(
            projected_xs,
            newshape=(-1, self.num_attention_heads, self.size_attention_heads),
        )
        if not use_position_embeddings:
            return hs
        return jax.vmap(apply_rotary_embeddings, in_axes=1, out_axes=1)(hs)

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        xs: Float32[Array, " seq_len size_layer"],
        enable_dropout: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> Float32[Array, " seq_len size_layer"]:
        xs_normalized = jax.vmap(self.norm)(xs)
        qs = self._compute_embeddings(
            xs_normalized,
            self.linear_q,
            use_position_embeddings=True,
        )
        ks = self._compute_embeddings(
            xs_normalized,
            self.linear_k,
            use_position_embeddings=True,
        )
        vs = self._compute_embeddings(xs_normalized, self.linear_v)
        attention_out = jax.vmap(compute_self_attention, in_axes=(1, 1, 1), out_axes=1)(
            qs, ks, vs
        )
        return jax.vmap(self.linear_o)(jax.lax.collapse(attention_out, 1, 3))
