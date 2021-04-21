FROM python:3.8-buster

COPY . /vad

WORKDIR /vad

RUN pip3 install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir -r requirements.txt

WORKDIR /vad/app

CMD ["python3", "./main.py"]