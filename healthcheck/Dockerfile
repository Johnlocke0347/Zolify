FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir requests

COPY entrypoint.py .

HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
  CMD python entrypoint.py || exit 1

CMD ["python", "entrypoint.py"]
