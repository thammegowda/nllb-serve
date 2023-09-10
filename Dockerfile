FROM python:3.8.14-slim-buster

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --upgrade pip

COPY ./requirements.txt ./requirements.txt

# Install package from source code
RUN pip3 install -r ./requirements.txt

EXPOSE 5001
