FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    procps \
    ca-certificates \
    netcat-openbsd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash miner

WORKDIR /app

COPY --chown=miner:miner docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

USER miner

HEALTHCHECK --interval=5s --timeout=3s --start-period=2s --retries=3 \
  CMD curl -f http://localhost:8080/ || exit 1

LABEL version="1.0"
LABEL description="Zolify Jolt-Native Prover Environment"

EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
