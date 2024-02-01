import jax
import pytest


@pytest.fixture(scope="session", autouse=True)
def jax_cpu():
    """Jax will use CPU only."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield
