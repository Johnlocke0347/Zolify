FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN sed -i 's/\r$//' docker/entrypoint.sh && chmod +x docker/entrypoint.sh

# Increased start-period and timeout to prevent premature 'unhealthy' status
HEALTHCHECK --interval=10s --timeout=10s --start-period=30s --retries=5 \
  CMD curl -f http://127.0.0.1:8080/ || exit 1

EXPOSE 8080

ENTRYPOINT ["/bin/bash", "/app/docker/entrypoint.sh"]
