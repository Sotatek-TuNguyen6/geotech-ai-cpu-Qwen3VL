FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV HUGGINGFACE_HUB_CACHE=/models/hf

# CPU threading an toàn (đỡ contention)
ENV OMP_NUM_THREADS=1

RUN pip install --no-cache-dir -U pip setuptools wheel \
 && pip install --no-cache-dir \
    "git+https://github.com/huggingface/transformers" \
 && pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
    accelerate safetensors pillow tqdm huggingface_hub

WORKDIR /app
COPY run_batch.py /app/run_batch.py
COPY CKK20241-C1-41.jpg /app/CKK20241-C1-41.jpg

CMD ["python","/app/run_batch.py"]
