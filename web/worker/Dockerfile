FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY web/worker/requirements.txt ./req.txt
RUN pip install --no-cache-dir -r req.txt

# main engine + config + worker package
COPY src/ ./src/
COPY config/ ./config/
COPY web/__init__.py ./web/__init__.py
COPY web/worker/ ./web/worker/

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "web.worker.worker"]
