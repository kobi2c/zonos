FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y python3 python3-pip python-is-python3 curl git espeak-ng ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install uv --break-system-packages

WORKDIR /app
COPY . ./

ENV UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu130"
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV MAX_JOBS=1

RUN uv pip install --system setuptools ninja psutil packaging torch torchvision torchaudio
RUN uv pip install --system -e ".[compile]"
