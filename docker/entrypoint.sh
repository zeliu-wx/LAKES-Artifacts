#!/usr/bin/env bash
set -euo pipefail

# Docker-in-Docker：启动镜像内部的 docker daemon（需 docker run --privileged）。
# SmartBugs 与此内部 daemon 通信，工具容器的临时目录挂载在同一文件系统命名空间，
# 规避 DooD（挂宿主 socket）下宿主 daemon 看不到容器内临时目录的路径不匹配问题。
if ! docker info >/dev/null 2>&1; then
  dockerd >/var/log/dockerd.log 2>&1 &
  for _ in $(seq 1 60); do
    docker info >/dev/null 2>&1 && break
    sleep 1
  done
  if ! docker info >/dev/null 2>&1; then
    echo "[error] 内部 dockerd 未能启动；真实执行（--execute）需要 docker run --privileged。" >&2
    echo "[error] 推荐流程本身不依赖 docker，可继续；以下为 dockerd 日志尾部：" >&2
    tail -n 30 /var/log/dockerd.log >&2 2>/dev/null || true
  fi
fi

exec toolrank "$@"
