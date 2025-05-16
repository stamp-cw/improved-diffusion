#!/bin/bash

# 优化后的安装脚本
set -euo pipefail  # 启用严格模式：出错退出、未定义变量报错、管道错误检测

function log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

function die() {
    log "错误: $1" >&2
    exit 1
}

log "=== 开始安装过程 ==="

# PART ONE: 使用conda安装mpi相关包
log "1. 正在安装conda依赖项 (mpi4py和openmpi)..."
conda install -y -c conda-forge mpi4py openmpi || die "conda安装失败"
log "conda依赖项安装完成"

# PART TWO: 使用pip安装requirements
REQUIREMENT_FILE="requirements.txt"
if [ -f "$REQUIREMENT_FILE" ]; then
    log "2. 检测到${REQUIREMENT_FILE}，正在安装Python依赖..."
    pip install --no-cache-dir -r "$REQUIREMENT_FILE" || die "pip依赖安装失败"
    log "Python依赖安装完成"
else
    log "警告: 未找到${REQUIREMENT_FILE}文件，跳过此步骤" >&2
fi

# PART THREE: 可编辑模式安装当前包
log "3. 正在以可编辑模式安装当前包..."
pip install --no-cache-dir -e . || die "可编辑模式安装失败"
log "可编辑模式安装完成"

log "=== 安装过程成功完成 ==="