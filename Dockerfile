FROM python:3.8.14-slim-buster

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
	
WORKDIR /app

COPY ./requirements.txt ./requirements.txt
COPY . /app

# Install package from source code
RUN pip install --upgrade pip && \
    pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install -r requirements.txt

EXPOSE 6060

# Command to start the server
CMD ["python", "-m", "nllb_serve"]