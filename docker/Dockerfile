FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    transformers \
    datasets \
    requests \
    fastapi \
    uvicorn \
    pydantic

RUN pip3 install zolify || echo "Zolify installation skipped"

COPY . .

RUN sed -i 's/\r$//' docker/entrypoint.sh && chmod +x docker/entrypoint.sh

HEALTHCHECK --interval=30s --timeout=20s --start-period=60s --retries=5 \
  CMD curl -f http://127.0.0.1:8080/ || exit 1

EXPOSE 8080
EXPOSE 8000

ENTRYPOINT ["/bin/bash", "/app/docker/entrypoint.sh"]
