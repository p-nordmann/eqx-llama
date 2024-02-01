FROM ghcr.io/nvidia/jax:nightly-pallas-2024-01-24

RUN apt update
RUN apt full-upgrade -y
RUN python3 -m pip install --upgrade pip

ENV XLA_PYTHON_CLIENT_PREALLOCATE="false"
ENV XLA_PYTHON_CLIENT_ALLOCATOR="platform"
ENV TF_FORCE_GPU_ALLOW_GROWTH="true"
ENV PYTHONDONTWRITEBYTECODE="1"
