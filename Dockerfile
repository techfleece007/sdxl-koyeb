FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install PyTorch first (matches torchvision) then the rest
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 torchvision==0.16.2+cu121 --find-links https://download.pytorch.org/whl/cu121/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
