#!/bin/bash
# LeRobot 数据上传系统卸载脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 停止并禁用服务
services=("lerobot-upload-daemon" "lerobot-web-dashboard")

for service in "${services[@]}"; do
    if sudo systemctl is-active --quiet "$service"; then
        log_info "停止服务: $service"
        sudo systemctl stop "$service"
    fi
    
    if sudo systemctl is-enabled --quiet "$service" 2>/dev/null; then
        log_info "禁用服务: $service"
        sudo systemctl disable "$service"
    fi
    
    if [ -f "/etc/systemd/system/$service.service" ]; then
        log_info "删除服务文件: $service.service"
        sudo rm "/etc/systemd/system/$service.service"
    fi
done

# 重载systemd
log_info "重载systemd..."
sudo systemctl daemon-reload

log_info "卸载完成！"
log_warn "注意: 日志文件和数据文件未被删除，如需清理请手动删除 upload_system_logs/ 目录"
