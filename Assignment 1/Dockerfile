FROM python:3.9-slim AS base

COPY ./requirements.txt /

RUN pip install --upgrade pip && \
	pip install --upgrade --no-cache-dir --requirement /requirements.txt && \
	pip install --no-cache-dir \
	autopep8 \
	isort

COPY .ignore/* /root/

WORKDIR /app

CMD ["bash"]
