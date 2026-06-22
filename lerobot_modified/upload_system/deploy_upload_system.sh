#!/bin/bash
# LeRobot 数据上传系统部署脚本
# 
# 功能：
# - 安装Python依赖
# - 配置systemd服务
# - 创建必要的目录
# - 检查配置文件

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

log_info "脚本目录: $SCRIPT_DIR"

log_info "项目根目录: $PROJECT_ROOT"

# 检查是否在虚拟环境中（支持venv和conda）
check_virtualenv() {
    # 检查 conda 环境
    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        log_info "检测到 Conda 环境: $CONDA_DEFAULT_ENV"
        PYTHON_PATH=$(which python)
        log_info "Python 路径: $PYTHON_PATH"
        return 0
    fi

    # 检查 venv 环境
    if [ -n "$VIRTUAL_ENV" ]; then
        log_info "虚拟环境已激活: $VIRTUAL_ENV"
        PYTHON_PATH="$VIRTUAL_ENV/bin/python"
        return 0
    fi

    # 尝试查找 conda 'lerobot' 环境（未激活时）
    if command -v conda &> /dev/null; then
        LEROBOT_ENV_PYTHON=$(conda run -n lerobot which python 2>/dev/null)
        if [ -n "$LEROBOT_ENV_PYTHON" ]; then
            log_info "发现 conda 'lerobot' 环境: $LEROBOT_ENV_PYTHON"
            PYTHON_PATH="$LEROBOT_ENV_PYTHON"
            return 0
        fi
    fi

    # 尝试查找并激活 .venv
    log_warn "未检测到虚拟环境（venv 或 conda）"
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        log_info "发现 .venv 虚拟环境，尝试激活..."
        source "$PROJECT_ROOT/.venv/bin/activate"
        PYTHON_PATH="$PROJECT_ROOT/.venv/bin/python"
        return 0
    fi

    log_error "未找到虚拟环境，请执行以下任一操作："
    log_error "  1. 激活 conda 环境: conda activate lerobot"
    log_error "  2. 创建 venv 环境: python -m venv .venv && source .venv/bin/activate"
    exit 1
}

# 安装Python依赖
install_dependencies() {
    log_info "安装Python依赖..."
    
    pip install --upgrade pip
    
    # 检查必要的包
    required_packages=("flask" "pyyaml" "psutil")
    for package in "${required_packages[@]}"; do
        if ! pip show "$package" > /dev/null 2>&1; then
            log_info "安装 $package..."
            pip install "$package"
        else
            log_info "$package 已安装"
        fi
    done
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."
    
    mkdir -p "$PROJECT_ROOT/upload_system_logs"
    mkdir -p "$PROJECT_ROOT/upload_system_logs/uploads"
    mkdir -p "$PROJECT_ROOT/config"
    
    log_info "目录创建完成"
}

# 检查配置文件
check_config() {
    log_info "检查配置文件..."
    
    config_file="$PROJECT_ROOT/upload_system/upload_config.yaml"
    if [ ! -f "$config_file" ]; then
        log_error "配置文件不存在: $config_file"
        log_info "请根据 upload_system/upload_config.yaml 示例创建配置文件"
        exit 1
    fi
    
    log_info "配置文件存在: $config_file"
    
    # 检查配置文件中的关键字段
    if ! grep -q "remote_ip:" "$config_file"; then
        log_warn "配置文件中缺少 remote_ip 字段"
    fi
}

# 安装systemd服务
install_systemd_services() {
    log_info "安装systemd服务..."
    
    # 检查是否有sudo权限
    if ! sudo -v > /dev/null 2>&1; then
        log_error "需要sudo权限来安装systemd服务"
        exit 1
    fi
    
    # 获取当前用户
    current_user=$(whoami)
    
    # 获取 Python 可执行文件的完整路径
    python_exec=$(which python)
    log_info "使用 Python: $python_exec"
    
    # 更新服务文件中的占位符
    python_bin_dir=$(dirname "$python_exec")

    for service_file in "$PROJECT_ROOT/upload_system/systemd"/*.service; do
        if [ -f "$service_file" ]; then
            service_name=$(basename "$service_file")

            # 创建临时文件并替换占位符
            temp_file=$(mktemp)
            sed -e "s|__USER__|$current_user|g" \
                -e "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" \
                -e "s|__PYTHON_BIN_DIR__|$python_bin_dir|g" \
                -e "s|__PYTHON_PATH__|$python_exec|g" \
                "$service_file" > "$temp_file"

            # 复制到systemd目录
            sudo cp "$temp_file" "/etc/systemd/system/$service_name"
            rm "$temp_file"

            log_info "已安装服务: $service_name"
        fi
    done
    
    # 重载systemd
    sudo systemctl daemon-reload
    log_info "systemd已重载"
}

# 启用并启动服务
enable_services() {
    log_info "启用并启动服务..."
    
    services=("lerobot-upload-daemon" "lerobot-web-dashboard")
    
    for service in "${services[@]}"; do
        log_info "启用服务: $service"
        sudo systemctl enable "$service"
        
        log_info "启动服务: $service"
        sudo systemctl start "$service"
        
        # 检查服务状态
        if sudo systemctl is-active --quiet "$service"; then
            log_info "✓ $service 运行中"
        else
            log_error "✗ $service 启动失败"
            log_info "查看日志: sudo journalctl -u $service -n 50"
        fi
    done
}

# 显示状态
show_status() {
    log_info "服务状态:"
    echo ""
    sudo systemctl status lerobot-upload-daemon --no-pager -l || true
    echo ""
    sudo systemctl status lerobot-web-dashboard --no-pager -l || true
}

# 显示访问信息
show_access_info() {
    echo ""
    log_info "==================================================="
    log_info "部署完成！"
    log_info "==================================================="
    echo ""
    log_info "Web控制面板访问地址:"
    log_info "  本地访问: http://localhost:5000"
    log_info "  远程访问: 通过VSCode SSH端口转发访问"
    echo ""
    log_info "查看服务状态:"
    log_info "  sudo systemctl status lerobot-upload-daemon"
    log_info "  sudo systemctl status lerobot-web-dashboard"
    echo ""
    log_info "查看日志:"
    log_info "  守护进程: tail -f $PROJECT_ROOT/upload_system_logs/upload_daemon.log"
    log_info "  Web面板: tail -f $PROJECT_ROOT/upload_system_logs/web_dashboard.log"
    log_info "  系统日志: sudo journalctl -u lerobot-upload-daemon -f"
    echo ""
    log_info "停止服务:"
    log_info "  sudo systemctl stop lerobot-upload-daemon"
    log_info "  sudo systemctl stop lerobot-web-dashboard"
    echo ""
    log_info "卸载服务:"
    log_info "  bash upload_system/uninstall_upload_system.sh"
    echo ""
}

# 主函数
main() {
    log_info "开始部署LeRobot数据上传系统..."
    echo ""
    
    # 进入项目根目录
    cd "$PROJECT_ROOT"
    
    # 执行部署步骤
    check_virtualenv
    install_dependencies
    create_directories
    check_config
    install_systemd_services
    enable_services
    
    echo ""
    show_status
    show_access_info
    
    log_info "部署完成！"
}

# 执行主函数
main
