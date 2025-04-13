FROM vllm/vllm-openai

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install torch tqdm transformers vllm

ENV NCCL_DEBUG=INFO
ENV CUDA_VISIBLE_DEVICES=0,1
ENV PYTHONUNBUFFERED=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

ENTRYPOINT ["python", "inference/inference.py"]