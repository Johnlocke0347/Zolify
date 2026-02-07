FROM python:3.11-slim
RUN pip install requests
COPY entrypoint.py /app/entrypoint.py
WORKDIR /app
CMD ["python", "entrypoint.py"]
