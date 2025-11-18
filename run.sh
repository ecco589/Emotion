#!/bin/bash

# 机器人情感交互系统 - Linux 一键运行脚本
# 自动检测依赖，缺失则安装，完整则直接运行

set -e  # 遇到错误立即退出

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }

# 检测Linux发行版
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
    elif [ -f /etc/debian_version ]; then
        DISTRO="debian"
    elif [ -f /etc/redhat-release ]; then
        DISTRO="rhel"
    else
        DISTRO="unknown"
    fi
    echo $DISTRO | tr '[:upper:]' '[:lower:]'
}

# 安装系统依赖
install_system_deps() {
    DISTRO=$(detect_distro)
    print_warning "检测到Linux发行版: $DISTRO"
    
    case $DISTRO in
        ubuntu|debian)
            print_warning "安装系统依赖（需要sudo权限）..."
            sudo apt-get update -qq
            sudo apt-get install -y \
                python3 python3-pip python3-venv \
                portaudio19-dev libasound2-dev \
                mpg123 fonts-wqy-microhei \
                build-essential libopencv-dev libsndfile1
            ;;
        centos|rhel|fedora)
            PKG_MGR=$(command -v dnf || command -v yum)
            print_warning "安装系统依赖（需要sudo权限）..."
            sudo $PKG_MGR install -y \
                python3 python3-pip \
                portaudio-devel alsa-lib-devel \
                mpg123 wqy-microhei-fonts \
                gcc gcc-c++ make opencv-devel libsndfile-devel
            ;;
        arch|manjaro)
            print_warning "安装系统依赖（需要sudo权限）..."
            sudo pacman -S --noconfirm \
                python python-pip \
                portaudio alsa-lib \
                mpg123 wqy-microhei \
                base-devel opencv libsndfile
            ;;
        *)
            print_error "未识别的Linux发行版: $DISTRO"
            print_warning "请手动安装依赖，然后重新运行此脚本"
            exit 1
            ;;
    esac
    print_success "系统依赖安装完成"
}

# 检查并安装Python依赖
check_and_install_python_deps() {
    # 禁用所有代理
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    unset all_proxy ALL_PROXY socks_proxy SOCKS_PROXY
    export http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY=""
    
    # 创建虚拟环境
    if [ ! -d "venv" ]; then
        print_warning "创建Python虚拟环境..."
        python3 -m venv venv
        print_success "虚拟环境创建完成"
    fi
    
    source venv/bin/activate
    
    # 升级pip（使用清华镜像源）
    python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple -q
    
    # 检查关键依赖
    MISSING=()
    python3 -c "import cv2" 2>/dev/null || MISSING+=("opencv-python")
    python3 -c "import pyaudio" 2>/dev/null || MISSING+=("pyaudio")
    python3 -c "import tensorflow" 2>/dev/null || MISSING+=("tensorflow")
    python3 -c "import keras" 2>/dev/null || MISSING+=("keras")
    python3 -c "import requests" 2>/dev/null || MISSING+=("requests")
    python3 -c "import PIL" 2>/dev/null || MISSING+=("pillow")
    python3 -c "import matplotlib" 2>/dev/null || MISSING+=("matplotlib")
    
    if [ ${#MISSING[@]} -gt 0 ]; then
        print_warning "检测到缺失依赖: ${MISSING[*]}"
        print_warning "正在安装（使用清华镜像源）..."
        python3 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
        print_success "Python依赖安装完成"
    else
        print_success "所有Python依赖已安装"
    fi
}

# 配置摄像头
configure_camera() {
    if [ ! -f "linux_config.txt" ]; then
        print_warning "检测摄像头设备..."
        CAMERAS=$(ls /dev/video* 2>/dev/null || echo "")
        
        if [ -z "$CAMERAS" ]; then
            print_error "未检测到摄像头设备"
            print_warning "请确保摄像头已连接到虚拟机"
            print_warning "VirtualBox: 设备 → USB → 选择摄像头"
            print_warning "VMware: 虚拟机 → 可移动设备 → 连接摄像头"
            exit 1
        fi
        
        echo "可用摄像头设备："
        for cam in $CAMERAS; do
            INDEX=$(echo $cam | sed 's/.*video//')
            echo "  $INDEX: $cam"
        done
        
        read -p "请选择摄像头索引 (0/1/2...): " camera_index
        echo "camera_index=$camera_index" > linux_config.txt
        print_success "摄像头配置已保存到 linux_config.txt"
    else
        print_success "使用已有配置文件: linux_config.txt"
    fi
}

# 配置权限
setup_permissions() {
    if ! groups | grep -q video; then
        print_warning "添加用户到video组（需要sudo）..."
        sudo usermod -a -G video $USER
        print_warning "请重新登录后权限生效，或运行: newgrp video"
    fi
    
    if ! groups | grep -q audio; then
        print_warning "添加用户到audio组（需要sudo）..."
        sudo usermod -a -G audio $USER
    fi
}

# 检查模型文件
check_models() {
    if [ ! -f "models/emotion_model.hdf5" ]; then
        print_error "未找到模型文件: models/emotion_model.hdf5"
        exit 1
    fi
    if [ ! -f "models/haarcascade_frontalface_default.xml" ]; then
        print_error "未找到人脸检测模型: models/haarcascade_frontalface_default.xml"
        exit 1
    fi
}

# 主函数
main() {
    echo "=========================================="
    echo "机器人情感交互系统 - Linux"
    echo "=========================================="
    echo ""
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "未找到 python3"
        print_warning "正在安装系统依赖..."
        install_system_deps
    fi
    
    # 检查系统依赖
    if ! dpkg -l | grep -q portaudio19-dev 2>/dev/null && ! rpm -q portaudio-devel &>/dev/null; then
        print_warning "系统依赖未安装，正在安装..."
        install_system_deps
    fi
    
    # 检查并安装Python依赖
    check_and_install_python_deps
    
    # 配置摄像头
    configure_camera
    
    # 配置权限
    setup_permissions
    
    # 检查模型文件
    check_models
    
    echo ""
    echo "=========================================="
    print_success "环境检查完成，启动程序..."
    echo "=========================================="
    echo ""
    
    # 激活虚拟环境并运行
    source venv/bin/activate
    python3 interactive_demo.py
}

main

