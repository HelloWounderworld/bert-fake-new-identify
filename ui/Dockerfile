FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

ARG UID
ARG USERNAME
ARG GID
ARG GROUPNAME
RUN groupadd -g ${GID} ${GROUPNAME} && useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install --assume-yes --no-install-recommends ca-certificates

# install nginx
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    nano \
    curl \
    wget \
    gnupg2 \
    las-release \
    ubuntu-keyring \
    && curl https://nginx.org/keys/nginx_signing.key | gpg --dearmor \
    | tee /usr/share/keyrings/nginx-archive-keyring.gpg >/dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/nginx-archive-keyring.gpg] http://nginx.org/packages/ubuntu `lsb_release -cs` nginx" \
    | tee /etc/apt/sources.list.d/nginx.list \
    && echo -e "Package: *\nPin: origin nginx.org\nPin: release o=nginx\nPin-Priority: 900\n" \
    | tee /etc/apt/preferences.d/99nginx \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y nginx

WORKDIR /etc/nginx/conf.d
COPY ./fake-news.conf .

WORKDIR /var
RUN mkdir www

WORKDIR /
# Install npm
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y npm

# install nodejs
WORKDIR /tmp
RUN wget https://nodejs.org/dist/v22.2.0/node-v22.2.0-linux-x64.tar.xz \
    && mkdir -p /usr/local/lib/nodejs \
    && tar -xJvf ./node-v22.2.0-linux-x64.tar.xz -C /usr/local/lib/nodejs

ENV PATH /usr/local/lib/nodejs/node-v22.2.0-linux-x64/bin:$PATH

RUN node --version && npm --version

RUN npm install -g @vue/cli

WORKDIR /home/${USERNAME}/ui

COPY . .

RUN npm install \
    && npm run build \
    && cp -r /home/${USERNAME}/ui1/dist/* /var/www/ \
    && service nginx start

CMD ["npm", "run", "serve"]
