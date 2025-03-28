FROM python:3.10-slim

RUN apt-get update \
    && pip install --no-cache-dir --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/CAG

COPY . /app/CAG

RUN pip3 install -r requirements.txt

EXPOSE 59488