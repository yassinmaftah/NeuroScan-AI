FROM python:3.11-slim

ARG REQ=requirements-tf.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ${REQ} requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi==0.115.5 uvicorn==0.32.1 python-multipart==0.0.17

COPY src/ src/

CMD ["python"]