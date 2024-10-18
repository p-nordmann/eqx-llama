FROM ghcr.io/nvidia/jax:jax

RUN apt update
RUN apt full-upgrade -y
RUN python3 -m pip install --upgrade pip

# Set environment variables.
ENV XLA_PYTHON_CLIENT_PREALLOCATE="false"
# ENV XLA_PYTHON_CLIENT_ALLOCATOR="platform"
ENV TF_FORCE_GPU_ALLOW_GROWTH="true"

ENV PYTHONDONTWRITEBYTECODE="1"
