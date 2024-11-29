FROM tiangolo/uvicorn-gunicorn:python3.11-slim

SHELL ["/bin/bash", "-c"]

ARG UID
ARG USERNAME
ARG GID
ARG GROUPNAME
RUN groupadd -g ${GID} ${GROUPNAME} && useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=off

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive \
    apt-get install --assume-yes --no-install-recommends \
    nano \
    sudo

WORKDIR /api

COPY ./requirements.txt ./

RUN python --version \
    python -m venv --help \
    python -m venv .venv \
    source .venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --trusted-host pypi.python.org -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "3000"]