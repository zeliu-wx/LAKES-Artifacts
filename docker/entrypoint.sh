#!/usr/bin/env bash
set -euo pipefail

# 校验 Docker socket 已挂载（真实执行依赖它拉起工具镜像）
if [ ! -S /var/run/docker.sock ]; then
  echo "[warn] /var/run/docker.sock 未挂载；--execute 真实执行将不可用。" >&2
  echo "[warn] 运行时请加 -v /var/run/docker.sock:/var/run/docker.sock" >&2
fi

exec toolrank "$@"
