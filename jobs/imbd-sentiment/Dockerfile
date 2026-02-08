FROM python:3.11-slim
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY train.py /app/train.py
WORKDIR /app
CMD ["python", "train.py"]
