FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects the PORT env var. --timeout 0 disables the worker timeout so
# long Gemini responses are not cut off mid-generation.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
