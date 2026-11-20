FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_OFFLINE=1 HOST=0.0.0.0 PORT=8080

# CPU-only torch keeps the image small; the model is tiny enough not to need a GPU.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install transformers sentencepiece protobuf safetensors fastapi uvicorn

# The converted model (run download_model.py locally first) is baked into the image.
COPY model ./model
COPY thinker.py server.py ./
COPY static ./static

# /app/data holds the thought history; on Fly it is a persistent volume.
EXPOSE 8080
CMD ["python", "server.py"]
