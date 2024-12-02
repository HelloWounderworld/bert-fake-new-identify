FROM ubuntu:22.04

# Vamos definir o shell padrao, de forma segura, que e representado pelo "-c"
SHELL ["/bin/bash", "-c"]

# Vamos automatizar algumas construcoes e instalacoes de pacotes do Ubuntu, de forma nao interativa
ARG DEBIAN_FRONTEND=noninteractive

# Atualiza os pacotes do sistema
RUN apt-get update\
    && apt-get upgrade -y \
    && apt-get install -y\
    nano \
    curl \
    wget \
    npm \
    ca-certificates \
    lsb-release \
    gnupg2 \
    ubuntu-keyring \
    && curl https://nginx.org/keys/nginx_signing.key | gpg --dearmor \
    | tee /usr/share/keyrings/nginx-archive-keyring.gpg >/dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/nginx-archive-keyring.gpg] http://nginx.org/packages/ubuntu `lsb_release -cs` nginx" \
    | tee /etc/apt/sources.list.d/nginx.list \
    && echo -e "Package: *\nPin: origin nginx.org\nPin: release o=nginx\nPin-Priority: 900\n" \
    | tee /etc/apt/preferences.d/99nginx \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y nginx \
    && service nginx start

# Isso ira instalar o NodeJs usando o wget
WORKDIR /tmp
RUN wget https://nodejs.org/dist/v22.2.0/node-v22.2.0-linux-x64.tar.xz \
    && mkdir -p /usr/local/lib/nodejs \
    && tar -xJvf ./node-v22.2.0-linux-x64.tar.xz -C /usr/local/lib/nodejs

ENV PATH /usr/local/lib/nodejs/node-v22.2.0-linux-x64/bin:$PATH

# Verifica as versões instaladas
RUN node --version && npm --version

# Instala Vue CLI globalmente
RUN npm install -g @vue/cli

# Define o diretorio de trabalho para frontend
WORKDIR /ui1

# Copie todos os arquivos do projeto backend
COPY . /ui1/

# Tentando instalar as dependencias do package.json
RUN npm install \
    && npm build \
    && npm install -D sass sass-loader

CMD ["npm", "run", "serve"]