FROM python:3.11-slim-bullseye
WORKDIR /app

# Update and upgrade system packages to their latest versions
RUN apt-get update && apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV PYTHONPATH=/app:$PYTHONPATH
CMD ["python", "/app/src/BaccusModel.py"]