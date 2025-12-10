# 机器人情感交互系统

实时情绪检测与智能对话系统，支持语音识别和情绪分析。

## 快速开始

```bash
chmod +x run.sh
./run.sh
```

## 操作说明

- **开始录音**: 按住空格键说话
- **停止录音**: 松开空格键
- **退出程序**: 按 Q 或 ESC 键

## 系统要求

- Linux (Ubuntu 18.04+)
- Python 3.7+
- 摄像头和麦克风

## 配置

编辑 `linux_config.txt` 设置摄像头索引：
```
camera_index=0
```

## 故障排除

**摄像头无法打开**:
```bash
sudo usermod -a -G video $USER  # 添加到video组
```

**音频问题**:
```bash
sudo usermod -a -G audio $USER  # 添加到audio组
```
