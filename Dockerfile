FROM python:3.11-slim

ARG SMARTBUGS_REF=89c16bb620c6bfb10e9c025f9372c9a80b1c5279

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    TOOLRANK_SMARTBUGS_DIR=/opt/smartbugs

# 系统依赖：git（克隆 smartbugs）、完整 Docker 引擎（Docker-in-Docker）、构建工具。
# DinD 必需：dockerd + containerd + iptables（容器网络）。SmartBugs 与镜像内部
# 的 dockerd 通信，临时目录挂载在同一文件系统命名空间内，规避 DooD 路径不匹配。
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl gnupg build-essential iptables \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        docker-ce docker-ce-cli containerd.io \
    && rm -rf /var/lib/apt/lists/*

# solc-select（多版本 solc）
RUN pip install solc-select

# SmartBugs：克隆 + poetry 安装到系统环境
RUN pip install poetry \
    && git clone https://github.com/smartbugs/smartbugs /opt/smartbugs \
    && cd /opt/smartbugs \
    && git checkout ${SMARTBUGS_REF} \
    && poetry config virtualenvs.in-project true \
    && poetry install --only main --no-interaction --no-root

# 叠加定制工具配置（mando-hgt / vulhunter 等）
COPY docker/smartbugs-tools/ /opt/smartbugs/tools/

# ToolRank 本体
WORKDIR /work
COPY . /work
RUN pip install -e .

# 内部 dockerd 的镜像/层存储；可用命名卷挂载以跨运行缓存已拉取的工具镜像
VOLUME /var/lib/docker

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
