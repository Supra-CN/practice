# syntax=docker/dockerfile:1
FROM python:3.7-alpine

WORKDIR /cph4/web

EXPOSE 5000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# setup alpine image
RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apk/repositories

RUN apk add --no-cache gcc musl-dev linux-headers

# COPY requirements.txt requirements.txt
COPY . .

RUN pip install -r requirements.txt
# EXPOSE 5000
# COPY . .

# Creates a non-root user with an explicit UID and adds permission to access the /cph4-web folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /cph4/web
USER appuser

# CMD ["flask", "run"]
# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["gunicorn", "--reload", "--bind", "0.0.0.0:5000", "app:app"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]