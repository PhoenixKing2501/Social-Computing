FROM python:3.9-slim AS base

RUN apt update && \
	apt upgrade -y && \
	apt install -y libgomp1 && \
	apt clean && \
	rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /

RUN pip install --upgrade pip && \
	pip install --upgrade --no-cache-dir --requirement /requirements.txt && \
	pip install --upgrade --no-cache-dir autopep8 isort

COPY .ignore/* /root/

WORKDIR /app

CMD ["bash"]
