FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CPU_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 INFERENCE_RUNTIME=onnx ONNX_PROFILE=default ONNX_FILENAME=best_unireplknet_a.onnx HF_HOME=/tmp/huggingface
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
COPY backend /app/backend
COPY tomatocare /app/tomatocare
# One worker: one model/session. No dataset, weights, secrets or training dependencies.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
