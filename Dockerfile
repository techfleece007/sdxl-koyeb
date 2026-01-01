FROM nvidia/cuda:12.1.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python + dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    libcudnn8 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages with longer timeout (heavy PyTorch + diffusers)
COPY requirements.txt .
RUN pip3 install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000

CMD ["./start.sh"]
