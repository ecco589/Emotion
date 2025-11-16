#!/bin/bash
cd "$(dirname "$0")"

echo "============================================================"
echo "机器人情感交互系统 - 启动中..."
echo "============================================================"
echo ""

# 使用系统Python
PYTHON_CMD="/usr/bin/python3"

# 检查Python
if [ ! -f "$PYTHON_CMD" ]; then
    echo "错误：未找到python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
$PYTHON_CMD -c "import cv2, numpy, keras, pyaudio, pynput, requests; print('✓ 所有依赖已安装')" 2>&1
if [ $? -ne 0 ]; then
    echo "错误：依赖检查失败，请确保所有依赖已安装"
    echo "安装命令：python3 -m pip install keras tensorflow numpy opencv-python scipy pillow pandas matplotlib h5py pyaudio pynput requests"
    echo ""
    echo "按任意键退出..."
    read -n 1
    exit 1
fi

echo ""
echo "启动程序..."
echo ""

# 运行程序
$PYTHON_CMD interactive_demo.py

# 如果程序退出，保持窗口打开
echo ""
echo "程序已退出"
echo "按任意键关闭窗口..."
read -n 1

