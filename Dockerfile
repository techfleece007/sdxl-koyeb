# Base image with CUDA
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install python and git
RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install PyTorch first from PyTorch links, then rest
RUN pip install --no-cache-dir \
    torch==2.1.1+cu121 torchvision==0.16.2+cu121 --find-links https://download.pytorch.org/whl/cu121/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

# Startup script
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000

CMD ["./start.sh"]
