FROM python:3.11-slim

# ------------------------------------------------------------
# System dependencies (OpenCV, GLib, etc.)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Hugging Face Spaces non-root user (UID 1000)
# ------------------------------------------------------------
RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR $HOME/app

# ------------------------------------------------------------
# Install Python dependencies
# Cài đặt PyTorch CPU trước để tiết kiệm dung lượng và tránh timeout build HF
# ------------------------------------------------------------
COPY --chown=user requirements.txt $HOME/app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# Copy source code and models
# ------------------------------------------------------------
COPY --chown=user . $HOME/app

# ------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------
ENV MODEL_PATH=models/best.pt \
    CONFIDENCE_THRESHOLD=0.25 \
    IMAGE_SIZE=640

# ------------------------------------------------------------
# Port (Hugging Face Spaces bắt buộc mở port 7860)
# ------------------------------------------------------------
EXPOSE 7860

# ------------------------------------------------------------
# Start FastAPI application
# ------------------------------------------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]