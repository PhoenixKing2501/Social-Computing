FROM python:3.9-slim AS base

RUN apt update && \
	apt upgrade -y && \
	apt install -y \
	libgomp1 \
	procps && \
	apt clean && \
	rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /

RUN pip install --upgrade pip && \
	pip install --upgrade --no-cache-dir autopep8 isort && \
	pip install --upgrade --no-cache-dir --requirement /requirements.txt

COPY .ignore/* /root/
COPY .ignore/toprc /root/.config/procps/toprc

WORKDIR /app

CMD ["bash"]
