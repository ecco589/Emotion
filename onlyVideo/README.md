# OnlyVideo - 视频情绪检测

仅使用摄像头的情绪检测程序。

## 运行方式

### 方法1：使用运行脚本（推荐）
```bash
./run.sh
```

### 方法2：手动激活虚拟环境
```bash
# 激活父目录的虚拟环境
source ../venv/bin/activate

# 运行程序
python emotions.py

# 退出虚拟环境
deactivate
```

### 方法3：直接使用虚拟环境的Python
```bash
../venv/bin/python emotions.py
```

## 配置

编辑 `emotions.py` 第13行：
- `USE_WEBCAM = True` - 使用摄像头
- `USE_WEBCAM = False` - 使用视频文件（`./demo/dinner.mp4`）

## 操作

- 按 `q` 键退出程序
