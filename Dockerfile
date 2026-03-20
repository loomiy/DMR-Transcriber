FROM python:3.13
#FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["python", "-u", "main.py"]
