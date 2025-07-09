FROM python:3.11-slim-bullseye
WORKDIR /app

# Update and upgrade system packages
RUN apt-get update && apt-get upgrade -y && \
    # 科学計算ライブラリのビルドに必要な依存関係をインストール
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gfortran \
    pkg-config \
    libatlas-base-dev && \
    # aptキャッシュをクリーンアップしてイメージサイズを小さく保つ
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV PYTHONPATH=/app:$PYTHONPATH
CMD ["python", "/app/src/BaccusModel.py"]