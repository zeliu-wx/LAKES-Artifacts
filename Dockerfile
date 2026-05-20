FROM python:3.11-slim

ARG SMARTBUGS_REF=89c16bb620c6bfb10e9c025f9372c9a80b1c5279

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    TOOLRANK_SMARTBUGS_DIR=/opt/smartbugs

# 系统依赖：git（克隆 smartbugs）、docker CLI（与宿主 socket 通信）、构建工具
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl gnupg build-essential \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# solc-select（多版本 solc）
RUN pip install solc-select

# SmartBugs：克隆 + poetry 安装到系统环境
RUN pip install poetry \
    && git clone https://github.com/smartbugs/smartbugs /opt/smartbugs \
    && cd /opt/smartbugs \
    && git checkout ${SMARTBUGS_REF} \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-root

# 叠加定制工具配置（mando-hgt / vulhunter 等）
COPY docker/smartbugs-tools/ /opt/smartbugs/tools/

# ToolRank 本体
WORKDIR /work
COPY . /work
RUN pip install -e .

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
