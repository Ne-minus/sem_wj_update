FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# # Install gnupg (required for apt-key)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gnupg2 \
#     && rm -rf /var/lib/apt/lists/*

# # Add NVIDIA's package repository (for NCCL)
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
#     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
#     apt-get update

# # Install system dependencies (NCCL and git)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git \
#     libnccl2 \
#     libnccl-dev \
#     && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip && \
    pip install torch tqdm transformers vllm==0.6.1

ENV NCCL_DEBUG=INFO
ENV CUDA_VISIBLE_DEVICES=0,1
ENV PYTHONUNBUFFERED=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

CMD ["python", "inference/inference.py"]