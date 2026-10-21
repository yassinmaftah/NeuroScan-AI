FROM python:3.11-slim

ARG REQ=requirements-tf.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ${REQ} requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

CMD ["python"]