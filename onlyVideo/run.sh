#!/bin/bash
# onlyVideo 运行脚本 - 使用父目录的虚拟环境

cd /home/ecco/Downloads/Emotion-linux
source venv/bin/activate
cd onlyVideo
python emotions.py
