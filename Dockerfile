FROM python:3.8.14-slim-buster

WORKDIR /app

RUN pip3 install --upgrade pip

COPY ./requirements.txt ./requirements.txt

# Install package from source code
RUN pip3 install -r ./requirements.txt

EXPOSE 5001
