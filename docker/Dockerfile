FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 as builder
ENV PYTHON_VERSION=3.8.5
ENV HOME="/workspace"
WORKDIR $HOME

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

# Read justification for this section in the next link
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ENV DISTRO=ubuntu2004
ENV ARCH=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$ARCH/3bf863cc.pub

RUN apt-get update && apt-get install -y \
  libssl-dev \
  openssl \
  make \
  gcc \
  git \
  jq \
  wget \
  curl \
  # Build dependencies
  bzip2 \
  lib32readline-dev \
  libffi-dev \
  zlib1g-dev \
  libbz2-dev \
  libsqlite3-dev \
  libgl1-mesa-glx \
  libopenblas-dev \
  && rm -rf /var/lib/apt/lists/* \
  && git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv

## Install Poetry
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

# to run poetry directly as soon as it's installed
ENV PATH="$POETRY_HOME/bin:$PATH"

## Install PyEnv
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
RUN pyenv install $PYTHON_VERSION \
  && pyenv global $PYTHON_VERSION \
  && ln -s $(which python3) /usr/local/bin/python

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && chmod 755 ${POETRY_HOME}/bin/poetry

COPY pyproject.toml /$HOME/pyproject.toml
COPY poetry.lock /$HOME/poetry.lock

# Install dependencies 
RUN poetry install --only main --no-root --no-ansi

## Inference image
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 as runtime
ENV HOME="/workspace"
WORKDIR $HOME

# copy the venv folder from builder image 
COPY --from=builder /workspace/.venv /workspace/.venv
COPY --from=builder /workspace/.pyenv /workspace/.pyenv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/workspace/.venv/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"

# Read justification for this section in the next link
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ENV DISTRO=ubuntu2004
ENV ARCH=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$ARCH/3bf863cc.pub

# Runtime dependencies
RUN apt-get update \
    && apt-get install -y \
    openssl \
    # Dependency for CBCT VTK
    libgl1-mesa-glx \
    libxrender1 \
    # Dependency for Minkowski Engine
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# install Minkowski Engine
RUN python -m pip install git+https://github.com/NVIDIA/MinkowskiEngine.git

COPY . /$HOME/

