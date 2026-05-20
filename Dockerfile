FROM python:3.11-slim

ARG SMARTBUGS_REF=89c16bb620c6bfb10e9c025f9372c9a80b1c5279

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    TOOLRANK_SMARTBUGS_DIR=/opt/smartbugs

# 系统依赖：git（克隆 smartbugs）、完整 Docker 引擎（Docker-in-Docker）、构建工具。
# DinD 必需：dockerd + containerd + iptables（容器网络）。SmartBugs 与镜像内部
# 的 dockerd 通信，临时目录挂载在同一文件系统命名空间内，规避 DooD 路径不匹配。
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl gnupg build-essential iptables default-jre-headless sudo \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        docker-ce docker-ce-cli containerd.io \
    && rm -rf /var/lib/apt/lists/*

# solc-select（多版本 solc）+ 预装常用版本（smartian 用本地 solc 编译合约；
# 其它版本运行时按需 solc-select install）
RUN pip install solc-select \
    && solc-select install 0.4.25 0.5.12 0.6.12 0.8.19 || true

# .NET 8 SDK（smartian 跑 Smartian.dll；run_smartian 的预检用 `dotnet --version` 需 SDK）
RUN curl -sSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh \
    && bash /tmp/dotnet-install.sh --channel 8.0 --install-dir /opt/dotnet \
    && rm /tmp/dotnet-install.sh

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

# Phase 2 特例工具：sailfish 瘦包装脚本（纯标准库；公开镜像 holmessherlock/sailfish
# 与 solc 均在运行时按需拉取/下载）
ENV TOOLRANK_SAILFISH_RUNNER=/work/docker/runners/run_sailfish.py

# Phase 2 特例工具：gptscan（按 runner 的 _run_gptscan 逻辑：用自带 venv 跑 src/main.py，
# Java 解析 src/jars，LLM 端点由运行时 -e 提供）。源码已随 COPY . /work 进镜像。
# 注意：requirements-docker.txt 已剔除 Ubuntu 系统泄漏包；falcon-analyzer 为 git 依赖、
# openai 为旧版 SDK——venv 安装设为非致命，失败也不阻断镜像构建（其余工具不受影响）。
ENV TOOLRANK_GPTSCAN_ROOT=/work/docker/vendor/gptscan
RUN python -m venv /work/docker/vendor/gptscan/.venv \
    && ( /work/docker/vendor/gptscan/.venv/bin/pip install -r /work/docker/vendor/gptscan/requirements-docker.txt \
         || echo "[warn] gptscan venv 部分依赖安装失败；gptscan 运行时可能需补依赖" )

# Phase 2 特例工具：securify2（按 runner 的 _run_securify2 逻辑：securifyjson.py 在 DinD 内
# 按合约 solc 版本 docker build securify:{ver} 再运行；securifyjson.py 调用 `sudo docker`，
# 故镜像装 sudo。源码已随 COPY . /work 进镜像）
ENV TOOLRANK_SECURIFY2_RUNNER=/work/docker/vendor/securify2/securifyjson.py

# Phase 2 特例工具：smartian（.NET 8 跑 Smartian.dll；run_smartian 用本地 solc 编译合约后
# 模糊测试）。构建产物与瘦包装脚本已随 COPY . /work 进镜像。INVARIANT 绕开 libicu 依赖。
ENV DOTNET_ROOT=/opt/dotnet \
    PATH=/opt/dotnet:${PATH} \
    DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 \
    TOOLRANK_SMARTIAN_RUNNER=/work/docker/runners/run_smartian.py

# 选一个默认 solc 版本，使裸 `solc` 可用（smartian 等的预检 `solc --version` 需要）。
# 运行时各工具仍可 `solc-select use <ver>` 按合约覆盖。
RUN solc-select use 0.5.12

# Smartian.dll 在构建机上把资源（src/Agent/*.bin 等）的绝对路径焊进了二进制。
# 用符号链接把原构建路径重定向到镜像内 vendored 副本，使写死路径在运行时可解析。
RUN mkdir -p /Users/liuze/Downloads/QuantifyX \
    && ln -sfn /work/docker/vendor/smartian /Users/liuze/Downloads/QuantifyX/Smartian

# 内部 dockerd 的镜像/层存储；可用命名卷挂载以跨运行缓存已拉取的工具镜像
VOLUME /var/lib/docker

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
