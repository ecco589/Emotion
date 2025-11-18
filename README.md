# 机器人情感交互系统 - Linux 版本

## 简介

基于深度学习的实时情绪检测与智能对话系统，支持Linux平台。

## 功能特性

    实时情绪检测（基于摄像头）
    语音识别（百度ASR）
    智能对话生成（豆包LLM）
    语音合成（百度TTS，度泽言音色）

## 快速开始

### 一键运行

```bash
# 下载代码后
chmod +x run.sh
./run.sh
```

脚本会自动：
- 检测并安装系统依赖
- 创建Python虚拟环境
- 安装Python依赖包（使用清华镜像源）
- 配置摄像头
- 运行程序

### 首次运行

1. **下载代码**
   ```bash
   # 从GitHub下载ZIP或使用git
   git clone https://github.com/ecco589/Emotion.git
   cd Emotion
   git checkout linux
   ```

2. **运行脚本**
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

3. **配置摄像头**
   - 脚本会自动检测摄像头设备
   - 选择摄像头索引（0/1/2...）
   - 配置保存在 `linux_config.txt`

4. **后续运行**
   - 直接运行 `./run.sh` 即可
   - 无需重新安装

## 系统要求

- **操作系统**: Linux (Ubuntu 18.04+, Debian 10+, CentOS 7+, Fedora 30+, Arch Linux)
- **Python**: 3.7+
- **内存**: 4GB RAM
- **摄像头**: USB摄像头或内置摄像头
- **麦克风**: 用于语音输入

## 摄像头配置

### 虚拟机中使用电脑前置摄像头

**VirtualBox**:
1. 插入USB摄像头
2. 菜单：设备 → USB → 选择摄像头设备

**VMware**:
1. 菜单：虚拟机 → 可移动设备 → 连接摄像头

### 手动配置

编辑 `linux_config.txt`:
```
camera_index=0  # 修改为你的摄像头索引
```

查看可用摄像头：
```bash
ls -l /dev/video*
```

## 故障排除

### 摄像头无法打开

1. 检查设备：`ls -l /dev/video*`
2. 检查权限：`groups`（确保在video组）
3. 添加到video组：`sudo usermod -a -G video $USER`，然后重新登录

### 依赖安装失败

脚本使用清华镜像源，如果仍有问题：
1. 检查网络连接
2. 确保代理已禁用
3. 手动安装：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 权限问题

```bash
# 添加到video组（摄像头）
sudo usermod -a -G video $USER

# 添加到audio组（音频）
sudo usermod -a -G audio $USER

# 重新登录后生效
```

## 操作说明

- **开始交互**: 按住空格键，开始说话
- **停止录音**: 松开空格键
- **退出程序**: 按 Q 键或 ESC 键

## 配置文件

- `linux_config.txt`: 摄像头配置
- `interactive_demo.py`: 主程序（API密钥在第39-44行）

## 许可证

详见 [LICENSE](LICENSE) 文件

## 联系方式

- GitHub: https://github.com/ecco589/Emotion
- Issues: https://github.com/ecco589/Emotion/issues
