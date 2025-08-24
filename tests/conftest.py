import jax
import pytest


def pytest_configure(config: pytest.Config):
    config.addinivalue_line("markers", "cpu: the test must be run on CPU")
    config.addinivalue_line("markers", "gpu: the test must be run on GPU")
    config.addinivalue_line("markers", "tpu: the test must be run on TPU")


def pytest_generate_tests(metafunc):
    configure_device_auto(metafunc)


def configure_device_auto(metafunc: pytest.Metafunc):
    if "device_auto" not in metafunc.fixturenames:
        return

    platforms_to_test = []
    if metafunc.definition.get_closest_marker("cpu"):
        platforms_to_test.append("cpu")
    if metafunc.definition.get_closest_marker("gpu"):
        platforms_to_test.append("gpu")
    if metafunc.definition.get_closest_marker("tpu"):
        platforms_to_test.append("tpu")
    if platforms_to_test:
        metafunc.parametrize("device_auto", platforms_to_test, indirect=True)


@pytest.fixture(scope="function", autouse=True)
def device_auto(request: pytest.FixtureRequest):
    requested_platform = getattr(request, "param", None)

    device = None
    if requested_platform is None:
        device = get_default_device()
    elif get_devices_safe(requested_platform):
        device = get_devices_safe(requested_platform)[0]

    if device is not None:
        with jax.default_device(device):
            yield
    else:
        pytest.skip(f"Requested device '{requested_platform}' not available.")


def get_default_device():
    return jax.devices()[0]


def get_devices_safe(platform: str):
    try:
        return jax.devices(platform)
    except RuntimeError:
        return []
