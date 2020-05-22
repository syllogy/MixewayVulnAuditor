FROM python:3.8-buster

LABEL maintainer="Grzegorz Siewruk <gsiewruk@gmail.com>"

WORKDIR ./app
ENV PYTHONPATH /app/src
ADD ./src/ ./src
COPY ./model/ ./model
COPY requirements.txt .

RUN apt-get update && \
    apt-get -y install python3-pandas

RUN pip install --upgrade --no-cache-dir -r requirements.txt

CMD ["python","/app/src/main/vuln_auditor_server.py"]

