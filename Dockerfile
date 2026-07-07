FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace/sglang/python

COPY tokenflow-sglang.tar.gz /tmp/

RUN mkdir -p /workspace/sglang && \
    cd /workspace/sglang && \
    tar -xzf /tmp/tokenflow-sglang.tar.gz

WORKDIR /workspace/sglang

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    unzip \
    build-essential \
    cmake \
    libopenmpi-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    setuptools-scm \
    wheel \
    packaging \
    build

RUN python3 -m pip install -e "/workspace/sglang/python[all]" -i https://mirrors.aliyun.com/pypi/simple/
RUN python3 -c "import sglang, sglang.launch_server; print(sglang.__file__)"
RUN python3 -m pip show sglang || true

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint
RUN chmod +x /usr/local/bin/docker-entrypoint

EXPOSE 30000
ENTRYPOINT ["docker-entrypoint"]
