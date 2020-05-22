FROM python:3.8-buster

LABEL maintainer="Grzegorz Siewruk <gsiewruk@gmail.com>"

COPY ./src/ ./app
COPY ./model/ ./app
COPY requirements.txt ./app
WORKDIR ./app
RUN ls -la

RUN apt-get update && \
    apt-get -y install python3-pandas

RUN pip install --upgrade --no-cache-dir -r requirements.txt

CMD ["python","/app/src/main/vuln_auditor_server.py"]

