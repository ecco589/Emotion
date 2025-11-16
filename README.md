# 真实交互Demo - 完整配置说明文档

本文档详细说明如何配置和自定义真实交互Demo程序的各个功能模块，方便后续与机器人头部结合（目前程序为mac版）。

---

## 目录

1. [API密钥配置](#1-api密钥配置)
2. [摄像头配置](#2-摄像头配置)
3. [录音配置](#3-录音配置)
4. [情绪检测配置](#4-情绪检测配置)
5. [机器人个性配置（Big Five）](#5-机器人个性配置big-five)
6. [LLM提示词配置](#6-llm提示词配置)
7. [语音合成（TTS）配置](#7-语音合成tts配置)
8. [界面显示配置](#8-界面显示配置)
9. [按键控制配置](#9-按键控制配置)
10. [其他配置项](#10-其他配置项)
11. [原Emotion程序readme](#11-原Emotion程序readme)

---

## 1. API密钥配置

### 1.1 位置
文件：`interactive_demo.py`  
行号：**第39-44行**

### 1.2 配置项说明

```python
# ==================== API配置 ====================
BAIDU_ASR_TOKEN = "..."  # 百度ASR的BCE签名token（备用，通常不使用）
BAIDU_ASR_API_KEY = "oYQuxRUoN93lwd847k782HOF"  # 百度ASR的API Key
BAIDU_ASR_SECRET_KEY = "UdbOQqdeWy6tkSso1ItspzMHiboxJX1Q"  # 百度ASR的Secret Key
DOUBAO_API_KEY = "0699bcd4-d849-4c1f-a8b5-38847b05531e"  # 豆包LLM的API Key
```

### 1.3 如何修改

1. **百度ASR API密钥**：
   - 访问 [百度智能云](https://cloud.baidu.com/)
   - 创建应用并获取API Key和Secret Key
   - 将获取的值替换到 `BAIDU_ASR_API_KEY` 和 `BAIDU_ASR_SECRET_KEY`

2. **豆包LLM API密钥**：
   - 访问 [火山引擎](https://www.volcengine.com/)
   - 获取豆包API的密钥
   - 将获取的值替换到 `DOUBAO_API_KEY`

### 1.4 注意事项
- 如果API密钥配置错误，语音识别和LLM生成功能将无法使用
- 程序会优先使用API_KEY和SECRET_KEY获取access_token，如果失败会尝试使用BCE签名token

---

## 2. 摄像头配置

### 2.1 位置
文件：`interactive_demo.py`  
函数：`init_emotion_module()`  
行号：**第95-109行**

### 2.2 配置项说明

```python
def init_emotion_module():
    # ...
    cap = cv2.VideoCapture(1)  # 第104行：MacBook前置摄像头
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # 第106行：默认摄像头（备用）
```

### 2.3 如何修改摄像头源

**修改第104行的摄像头索引**：

```python
# 示例1：使用外接USB摄像头（通常是索引2或更高）
cap = cv2.VideoCapture(2)  # 改为2、3、4等，根据实际情况调整

# 示例2：使用默认摄像头
cap = cv2.VideoCapture(0)

# 示例3：使用MacBook前置摄像头
cap = cv2.VideoCapture(1)

# 示例4：使用摄像头设备路径（Linux）
cap = cv2.VideoCapture("/dev/video0")
```

### 2.4 如何查找可用的摄像头索引

在Python中运行以下代码来检测可用的摄像头：

```python
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"摄像头 {i} 可用")
        cap.release()
```

### 2.5 注意事项
- 如果指定的摄像头索引无法打开，程序会自动尝试使用索引0（默认摄像头）
- 外接摄像头通常需要先连接并确保系统已识别
- macOS上可能需要授予摄像头权限

---

## 3. 录音配置

### 3.1 位置
文件：`interactive_demo.py`  
函数：`record_audio()`  
行号：**第253-299行**

### 3.2 配置项说明

```python
def record_audio():
    # ...
    CHUNK = 1024          # 第260行：音频缓冲区大小
    FORMAT = pyaudio.paInt16  # 第261行：音频格式（16位整数）
    CHANNELS = 1          # 第262行：声道数（单声道）
    RATE = 16000          # 第263行：采样率（16kHz）
```

### 3.3 如何修改录音参数

**修改采样率**（第263行）：
```python
RATE = 16000  # 16kHz（推荐，百度ASR要求）
# RATE = 8000   # 8kHz（较低质量）
# RATE = 44100  # 44.1kHz（高质量，但可能不被ASR支持）
```

**修改声道数**（第262行）：
```python
CHANNELS = 1  # 单声道（推荐，百度ASR要求）
# CHANNELS = 2  # 立体声（不推荐，ASR通常只需要单声道）
```

**修改缓冲区大小**（第260行）：
```python
CHUNK = 1024  # 默认值，通常不需要修改
# CHUNK = 512   # 更小的缓冲区，延迟更低但CPU占用更高
# CHUNK = 2048  # 更大的缓冲区，延迟更高但CPU占用更低
```

### 3.4 注意事项
- 百度ASR要求音频格式为：16kHz采样率、单声道、WAV格式
- 修改采样率或声道数可能导致语音识别失败
- 缓冲区大小影响录音延迟和CPU占用，建议保持默认值

---

## 4. 情绪检测配置

### 4.1 位置
文件：`interactive_demo.py`  
行号：**第46-55行**（全局变量）和 **第191-251行**（检测函数）

### 4.2 配置项说明

```python
# 全局变量配置（第46-55行）
emotion_model_path = './models/emotion_model.hdf5'  # 情绪模型路径
emotion_offsets = (20, 40)  # 第53行：人脸检测偏移量
frame_window = 10  # 第54行：情绪平滑窗口大小

# 检测函数中的配置（第196-201行）
faces = face_cascade.detectMultiScale(
    gray_image, 
    scaleFactor=1.1,      # 图像缩放因子
    minNeighbors=5,        # 最小邻居数
    minSize=(30, 30),      # 最小人脸尺寸
    flags=cv2.CASCADE_SCALE_IMAGE
)
```

### 4.3 如何修改

**修改情绪模型路径**（第48行）：
```python
emotion_model_path = './models/emotion_model.hdf5'  # 默认路径
# emotion_model_path = './models/your_custom_model.hdf5'  # 自定义模型
```

**修改人脸检测偏移量**（第53行）：
```python
emotion_offsets = (20, 40)  # (x偏移, y偏移)，用于扩大检测区域
# emotion_offsets = (30, 50)  # 更大的偏移，包含更多背景
# emotion_offsets = (10, 20)  # 更小的偏移，更精确的人脸区域
```

**修改情绪平滑窗口**（第54行）：
```python
frame_window = 10  # 使用最近10帧的情绪进行平滑
# frame_window = 5   # 更小的窗口，情绪变化更敏感
# frame_window = 20  # 更大的窗口，情绪变化更平滑
```

**修改人脸检测参数**（第196-201行）：
```python
faces = face_cascade.detectMultiScale(
    gray_image, 
    scaleFactor=1.1,      # 1.1-1.3之间，越小检测越慢但越准确
    minNeighbors=5,       # 3-6之间，越大误检越少但可能漏检
    minSize=(30, 30),     # 最小人脸尺寸，根据摄像头分辨率调整
    flags=cv2.CASCADE_SCALE_IMAGE
)
```

**修改情绪置信度阈值**（第244行）：
```python
if emotion_probability < 0.4:  # 置信度低于0.4时使用neutral
    emotion_mode = "neutral"
    emotion_probability = 0.5
# 可以修改为：
# if emotion_probability < 0.5:  # 更严格的阈值
# if emotion_probability < 0.3:  # 更宽松的阈值
```

### 4.4 注意事项
- 情绪模型文件必须存在，否则程序无法启动
- 平滑窗口大小影响情绪检测的实时性和稳定性
- 人脸检测参数需要根据实际使用场景调整

---

## 5. 机器人个性配置（Big Five）

### 5.1 位置
文件：`interactive_demo.py`  
行号：**第67-74行**

### 5.2 配置项说明

```python
robot_personality = {
    "openness": "medium",          # 开放性：中（不推荐新活动）
    "conscientiousness": "medium",  # 尽责性：中（不提醒）
    "extraversion": "medium",      # 外向性：中（1-2句话）
    "agreeableness": "high",       # 宜人性：高（说共情的话）
    "neuroticism": "low"           # 神经质：低（情绪程度波动小）
}
```

### 5.3 如何修改个性参数

每个参数可以设置为：`"low"`、`"medium"`、`"high"`

**示例1：创建一个"活泼外向型"机器人**：
```python
robot_personality = {
    "openness": "high",           # 高开放性：喜欢推荐新活动
    "conscientiousness": "low",    # 低尽责性：不提醒任务
    "extraversion": "high",       # 高外向性：说更多话
    "agreeableness": "high",       # 高宜人性：说共情的话
    "neuroticism": "low"           # 低神经质：情绪稳定
}
```

**示例2：创建一个"严谨负责型"机器人**：
```python
robot_personality = {
    "openness": "low",            # 低开放性：不推荐新活动
    "conscientiousness": "high",   # 高尽责性：会提醒任务
    "extraversion": "low",         # 低外向性：话少
    "agreeableness": "medium",     # 中宜人性：适度共情
    "neuroticism": "low"           # 低神经质：情绪稳定
}
```

**示例3：创建一个"敏感共情型"机器人**：
```python
robot_personality = {
    "openness": "medium",
    "conscientiousness": "medium",
    "extraversion": "medium",
    "agreeableness": "high",       # 高宜人性：非常共情
    "neuroticism": "high"          # 高神经质：情绪波动大
}
```

### 5.4 参数说明

- **openness（开放性）**：影响是否推荐新活动、新想法
- **conscientiousness（尽责性）**：影响是否提醒任务、关注细节
- **extraversion（外向性）**：影响回复的长度和活跃程度
- **agreeableness（宜人性）**：影响共情程度和友好程度
- **neuroticism（神经质）**：影响情绪波动的程度

### 5.5 注意事项
- 个性参数会影响LLM生成的回复风格
- 修改后需要重新运行程序才能生效
- 这些参数会传递给LLM的system prompt（见第6节）

---

## 6. LLM提示词配置

### 6.1 位置
文件：`interactive_demo.py`  
函数：`call_llm()`  
行号：**第598-673行**

### 6.2 配置项说明

#### 6.2.1 System Prompt（系统提示词）
位置：**第606-644行**

这是定义机器人角色和行为的核心提示词，包含：
- 机器人类型描述
- Big Five个性参数
- Appraisal评估步骤
- 行为生成规则
- 输出格式要求

#### 6.2.2 User Prompt（用户提示词）
位置：**第646-657行**

这是每次调用时传入的用户数据提示词，包含：
- 当前用户语音
- 用户情绪和置信度
- 记忆上下文
- 任务说明

#### 6.2.3 LLM API参数
位置：**第659-673行**

```python
url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
data = {
    "model": "deepseek-v3-1-terminus",  # 第666行：使用的模型
    "messages": [...],
    "max_completion_tokens": 100,        # 第671行：最大输出token数
    "temperature": 0.3                   # 第672行：温度参数（0-1）
}
```

### 6.3 如何修改

#### 修改机器人类型（第606行）：
```python
system_prompt = f"""你是"温和共情型"机器人。请严格按照以下规则工作，最后只输出3行结果。
# 可以改为：
# system_prompt = f"""你是"活泼外向型"机器人。...
# system_prompt = f"""你是"严谨负责型"机器人。...
```

#### 修改Appraisal评估步骤（第616-621行）：
```python
【Appraisal评估步骤】
请按照以下步骤评估事件：
1. 相关性检查：事件是否与机器人目标相关？（用户的情感状态）
2. 效价评估：事件的情感效价（positive/negative/sarcasm/neutral）
   - 注意：用户语音的字面意思和语气可能不一致（如反讽）
3. 应对潜力：机器人能否缓解负面情绪？如何回应？
```

可以添加更多评估步骤或修改评估逻辑。

#### 修改行为生成规则（第623-631行）：
```python
【行为生成规则】
基于评估结果和个性参数生成：
- 情感状态：根据Appraisal结果和用户情绪生成你的情感
  - 用户情绪置信度≥0.6：按用户情绪生成你的情感（共情）
  - 用户情绪置信度<0.6：默认neutral:0.7
- 行动选择：根据个性参数和记忆上下文选择回应方式
  - 宜人性高：说共情的话
  - 外向性中：1-2句话
  - 考虑用户偏好（如喜欢幽默）
```

可以修改情感生成逻辑和行动选择规则。

#### 修改输出格式（第633-643行）：
```python
【输出格式】
只输出3行，每行一个值：
情绪标签
情绪程度
回复文字

示例：
neutral
0.7
你好呀，有什么我能帮到你的吗？

重要：只输出3行，不要输出推理过程。"""
```

如果需要不同的输出格式，可以修改这部分。

#### 修改LLM模型（第666行）：
```python
"model": "deepseek-v3-1-terminus",  # 当前使用的模型
# 可以改为其他支持的模型，具体取决于豆包API支持哪些模型
```

#### 修改最大输出长度（第671行）：
```python
"max_completion_tokens": 100,  # 当前值
# "max_completion_tokens": 200,  # 允许更长的回复
# "max_completion_tokens": 50,   # 限制回复长度
```

#### 修改温度参数（第672行）：
```python
"temperature": 0.3  # 当前值（较低，更确定性）
# "temperature": 0.7  # 中等（更创造性）
# "temperature": 0.1  # 很低（非常确定性）
# "temperature": 1.0  # 高（非常创造性）
```

### 6.4 创建新的性格模型

要创建新的性格模型，需要：

1. **修改robot_personality**（见第5节）
2. **修改system_prompt中的机器人类型描述**（第606行）
3. **修改行为生成规则**（第623-631行），使其符合新性格
4. **可选：修改Appraisal评估步骤**（第616-621行）

**示例：创建"幽默风趣型"机器人**

```python
# 1. 修改个性配置（第67-74行）
robot_personality = {
    "openness": "high",
    "conscientiousness": "low",
    "extraversion": "high",
    "agreeableness": "high",
    "neuroticism": "low"
}

# 2. 修改system_prompt（第606行）
system_prompt = f"""你是"幽默风趣型"机器人。请严格按照以下规则工作，最后只输出3行结果。

【个性参数化（Big Five）】
你的个性参数：
- 开放性（Openness）：{robot_personality['openness']} - 喜欢推荐新活动，富有创意
- 尽责性（Conscientiousness）：{robot_personality['conscientiousness']} - 不提醒任务，轻松随意
- 外向性（Extraversion）：{robot_personality['extraversion']} - 话多，喜欢互动
- 宜人性（Agreeableness）：{robot_personality['agreeableness']} - 非常友好，喜欢开玩笑
- 神经质（Neuroticism）：{robot_personality['neuroticism']} - 情绪稳定，乐观

【行为生成规则】
- 情感状态：根据用户情绪生成相应的情感，但保持乐观基调
- 行动选择：
  - 在回复中加入适当的幽默元素
  - 使用轻松、友好的语气
  - 可以适当使用网络用语或表情符号
  - 保持1-3句话的长度

【输出格式】
只输出3行，每行一个值：
情绪标签
情绪程度
回复文字（可以包含幽默元素）

示例：
happy
0.8
哈哈，听起来不错！有什么我能帮你的吗？😊
"""
```

### 6.5 注意事项
- 修改system_prompt后需要重新运行程序
- 确保输出格式保持一致（3行：情绪标签、情绪程度、回复文字）
- temperature参数影响回复的创造性和一致性
- max_completion_tokens不要设置太小，否则可能截断回复

---

## 7. 语音合成（TTS）配置

### 7.1 位置
文件：`interactive_demo.py`  
函数：`text_to_speech()`  
行号：**第1069-1153行**

### 7.2 配置项说明

```python
def text_to_speech(text, access_token=None):
    # ...
    params_list = [
        f"tex={tex_encoded}",
        f"tok={access_token}",
        f"cuid={cuid}",
        "ctp=1",           # 客户端类型（固定值）
        "lan=zh",          # 语言（固定值，中文）
        "spd=5",           # 第1108行：语速（0-15）
        "pit=5",           # 第1109行：音调（0-15）
        "vol=5",           # 第1110行：音量（0-15）
        "per=4193",        # 第1111行：音色（度泽言）
        "aue=3"            # 第1112行：音频格式（3=mp3）
    ]
```

### 7.3 如何修改TTS参数

#### 修改音色（第1111行）

**大模型音库（推荐）**：
```python
"per=4193",  # 度泽言（当前使用）
# "per=4189",  # 度涵竹
# "per=4194",  # 度嫣然
# "per=4195",  # 度怀安
# "per=4196",  # 度清影
# "per=4197",  # 度沁遥
# "per=20100", # 度小粤
# "per=20101", # 度晓芸
# "per=4257",  # 四川小哥
# "per=4132",  # 度阿闽
# "per=4139",  # 度小蓉
# "per=5977",  # 台媒女声
# "per=4007",  # 度小台
# "per=4150",  # 度湘玉
# "per=4134",  # 度阿锦
# "per=4172",  # 度筱林
```

**臻品音库**：
```python
# "per=4003",  # 度逍遥（臻品）
# "per=4106",  # 度博文
# "per=4115",  # 度小贤
# "per=4119",  # 度小鹿
# "per=4105",  # 度灵儿
# "per=4117",  # 度小乔
# "per=4100",  # 度小雯
# "per=4103",  # 度米朵
# "per=4144",  # 度姗姗
# "per=4278",  # 度小贝
# "per=4143",  # 度清风
# "per=4140",  # 度小新
# "per=4129",  # 度小彦
# "per=4149",  # 度星河
# "per=4254",  # 度小清
# "per=4206",  # 度博文
# "per=4226",  # 南方
```

**精品音库**：
```python
# "per=5003",  # 度逍遥（精品）
# "per=5118",  # 度小鹿
# "per=106",   # 度博文
# "per=110",   # 度小童
# "per=111",   # 度小萌
# "per=103",   # 度米朵
# "per=5",     # 度小娇
```

**基础音库**：
```python
# "per=1",     # 度小宇
# "per=0",     # 度小美
# "per=3",     # 度逍遥（基础）
# "per=4",     # 度丫丫
```

#### 修改语速（第1108行）
```python
"spd=5",  # 当前值（中等语速）
# "spd=0",   # 最慢
# "spd=15",  # 最快
# "spd=3",   # 较慢
# "spd=7",   # 较快
```

#### 修改音调（第1109行）
```python
"pit=5",  # 当前值（中等音调）
# "pit=0",   # 最低
# "pit=15",  # 最高
# "pit=3",   # 较低
# "pit=7",   # 较高
```

#### 修改音量（第1110行）
```python
"vol=5",  # 当前值（中等音量）
# "vol=0",   # 最小（注意：0不是无声，是最小音量）
# "vol=15",  # 最大（大模型音库支持0-15）
# "vol=3",   # 较小
# "vol=7",   # 较大
```

#### 修改音频格式（第1112行）
```python
"aue=3",  # mp3格式（当前使用，推荐）
# "aue=4",  # pcm-16k/24k格式
# "aue=5",  # pcm-8k格式
# "aue=6",  # wav格式（内容同pcm-16k/24k）
```

### 7.4 修改音频文件保存路径（第1131行）

```python
audio_filename = "temp_tts_output.mp3"  # 当前值
# audio_filename = "./audio/tts_output.mp3"  # 保存到audio目录
# audio_filename = f"tts_{int(time.time())}.mp3"  # 使用时间戳命名
```

### 7.5 修改音频清理延迟（第1430行）

```python
time.sleep(35)  # 等待35秒后删除临时文件
# time.sleep(60)  # 等待60秒（如果音频较长）
# time.sleep(20)  # 等待20秒（如果音频较短）
```

### 7.6 注意事项
- 不同音库的音量范围可能不同（基础音库0-9，精品/大模型音库0-15）
- 修改音频格式后，可能需要修改播放函数以支持新格式
- 音色ID需要参考百度TTS官方文档的最新列表
- 语速、音调、音量的最佳值需要根据实际使用场景调整

---

## 8. 界面显示配置

### 8.1 位置
文件：`interactive_demo.py`  
行号：**第87-91行**（全局变量）和 **第1450-1560行**（显示函数）

### 8.2 配置项说明

#### 8.2.1 显示超时时间（第90-91行）
```python
display_timeout = 0  # 机器人回复显示超时时间
user_display_timeout = 0  # 用户语音显示超时时间
```

#### 8.2.2 显示区域位置（第1482-1492行）
```python
# 用户语音显示区域（底部，固定高度60像素）
user_y_start = frame_height - 60
user_y_end = frame_height - 10

# 机器人回复显示区域
if show_user_voice:
    robot_y_start = frame_height - 160
    robot_y_end = frame_height - 70
else:
    robot_y_start = frame_height - 100
    robot_y_end = frame_height - 10
```

#### 8.2.3 字体配置（第1461-1476行）
```python
try:
    font_path = "/System/Library/Fonts/PingFang.ttc"  # macOS中文字体
    font_large = ImageFont.truetype(font_path, 24)    # 大字体
    font_small = ImageFont.truetype(font_path, 20)    # 小字体
    font_medium = ImageFont.truetype(font_path, 22)   # 中等字体
except:
    # 备用字体
    font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
    # ...
```

#### 8.2.4 背景颜色（第1344-1354行）
```python
if user_voice_display == "未识别到语音":
    bg_color = (60, 30, 30)  # 红色背景（识别失败）
else:
    bg_color = (30, 60, 30)  # 绿色背景（识别成功）

# 机器人回复背景
cv2.rectangle(overlay, (10, robot_y_start), (frame_width - 10, robot_y_end), (30, 30, 60), -1)
```

### 8.3 如何修改

#### 修改显示超时时间（第1376行和第1209行）
```python
# 用户语音显示时间（第1376行）
user_display_timeout = time.time() + 30  # 显示30秒
# user_display_timeout = time.time() + 60  # 显示60秒
# user_display_timeout = time.time() + 10  # 显示10秒

# 机器人回复显示时间（第1209行，在update_memory函数中）
display_timeout = time.time() + 10  # 显示10秒
# display_timeout = time.time() + 20  # 显示20秒
# display_timeout = time.time() + 5   # 显示5秒
```

#### 修改显示区域位置和大小（第1482-1492行）
```python
# 用户语音显示区域
user_y_start = frame_height - 60   # 距离底部60像素开始
user_y_end = frame_height - 10     # 距离底部10像素结束（高度50像素）
# 可以修改为：
# user_y_start = frame_height - 80   # 更大的显示区域
# user_y_end = frame_height - 10

# 机器人回复显示区域
if show_user_voice:
    robot_y_start = frame_height - 160  # 用户语音上方
    robot_y_end = frame_height - 70
else:
    robot_y_start = frame_height - 100
    robot_y_end = frame_height - 10
```

#### 修改字体大小（第1464-1466行）
```python
font_large = ImageFont.truetype(font_path, 24)   # 大字体24号
font_small = ImageFont.truetype(font_path, 20)   # 小字体20号
font_medium = ImageFont.truetype(font_path, 22)  # 中等字体22号
# 可以修改为：
# font_large = ImageFont.truetype(font_path, 32)   # 更大的字体
# font_small = ImageFont.truetype(font_path, 16)   # 更小的字体
```

#### 修改字体路径（第1463行，适用于Linux/Windows）
```python
# macOS
font_path = "/System/Library/Fonts/PingFang.ttc"

# Linux（示例）
# font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"  # 中文字体

# Windows（示例）
# font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
# font_path = "C:/Windows/Fonts/msyh.ttc"    # 微软雅黑
```

#### 修改背景颜色（第1344-1354行）
```python
# 用户语音背景色（BGR格式）
if user_voice_display == "未识别到语音":
    bg_color = (60, 30, 30)   # 红色（B, G, R）
else:
    bg_color = (30, 60, 30)   # 绿色（B, G, R）

# 可以修改为：
# bg_color = (30, 30, 60)   # 蓝色
# bg_color = (60, 60, 30)   # 黄色
# bg_color = (40, 40, 40)   # 灰色

# 机器人回复背景色（第1354行）
cv2.rectangle(overlay, (10, robot_y_start), (frame_width - 10, robot_y_end), (30, 30, 60), -1)
# (30, 30, 60) 是蓝色背景，可以修改为其他颜色
```

#### 修改文字颜色（第1381-1390行）
```python
# 用户语音文字颜色（RGB格式）
if user_voice_display == "未识别到语音":
    text_color = (255, 150, 150)  # 浅红色
else:
    text_color = (200, 255, 200)  # 浅绿色

# 机器人回复文字颜色（第1403行和第1429行）
draw.text((20, robot_y_start + 10), emotion_text, fill=(255, 255, 255), font=font_medium)  # 白色
draw.text((20, y_pos), response_lines[i], fill=(200, 200, 255), font=font_small)  # 浅蓝色
```

#### 修改窗口标题（第1232行）
```python
cv2.namedWindow('情感交互系统', cv2.WINDOW_NORMAL)
# 可以修改为：
# cv2.namedWindow('我的机器人', cv2.WINDOW_NORMAL)
```

### 8.4 注意事项
- 字体路径需要根据操作系统调整
- 颜色值使用BGR格式（OpenCV）或RGB格式（PIL），注意区分
- 显示区域不要超出画面范围
- 字体大小需要根据显示区域大小调整

---

## 9. 按键控制配置

### 9.1 位置
文件：`interactive_demo.py`  
行号：**第1087-1105行**（按键回调函数）和 **第1300-1308行**（主循环中的按键检测）

### 9.2 配置项说明

```python
def on_key_press(key):
    """按键按下回调"""
    global space_pressed
    try:
        if key == keyboard.Key.space:  # 第1091行：空格键
            space_pressed = True

def on_key_release(key):
    """按键释放回调"""
    global space_pressed
    try:
        if key == keyboard.Key.space:  # 第1100行：空格键
            space_pressed = False
        elif key == keyboard.Key.esc:  # 第1102行：ESC键退出
            return False
```

### 9.3 如何修改按键

#### 修改录音按键（第1091行和第1100行）
```python
# 将空格键改为其他按键
if key == keyboard.Key.space:  # 当前：空格键
# 可以改为：
# if key == keyboard.Key.enter:     # Enter键
# if key == keyboard.Key.shift:      # Shift键
# if key == keyboard.Key.ctrl:       # Ctrl键
# if key == keyboard.Key.alt:        # Alt键
# if key == keyboard.KeyCode.from_char('r'):  # R键
```

#### 修改退出按键（第1102行和第1343行）
```python
# 方法1：在on_key_release函数中（第1102行）
elif key == keyboard.Key.esc:  # ESC键
    return False

# 方法2：在主循环中（第1343行）
if key == ord('q') or key == ord('Q'):  # Q键
    should_exit = True

# 可以修改为：
# elif key == keyboard.KeyCode.from_char('x'):  # X键退出
# if key == ord('e') or key == ord('E'):  # E键退出
```

#### 添加新的按键功能

在主循环中添加新的按键检测（第1305-1308行附近）：

```python
key = cv2.waitKey(1) & 0xFF
current_key_state = (key == ord(' ') or key == 32)
if key == ord('q') or key == ord('Q'):
    should_exit = True
# 添加新按键：
elif key == ord('s') or key == ord('S'):
    # 执行某个功能，例如保存截图
    cv2.imwrite(f"screenshot_{int(time.time())}.jpg", frame)
    print("已保存截图")
```

### 9.4 注意事项
- 使用pynput时，按键检测更实时，但需要辅助功能权限（macOS）
- 使用OpenCV的waitKey时，按键检测可能不够实时
- 某些系统按键（如Cmd、Win）可能无法检测
- 修改按键后需要重新运行程序

---

## 10. 其他配置项

### 10.1 情绪模型路径配置

**位置**：第48行
```python
emotion_model_path = './models/emotion_model.hdf5'
```

**如何修改**：
```python
emotion_model_path = './models/your_model.hdf5'  # 使用自定义模型
emotion_model_path = '/path/to/model.hdf5'       # 使用绝对路径
```

### 10.2 人脸检测模型路径配置

**位置**：第100行
```python
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
```

**如何修改**：
```python
# 使用其他人脸检测模型
face_cascade = cv2.CascadeClassifier('./models/haarcascade_profileface.xml')  # 侧脸检测
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt.xml')  # 替代模型
```

### 10.3 临时文件路径配置

**位置**：第303行（录音文件）和第1131行（TTS文件）
```python
filename = "temp_voice.wav"  # 录音文件
audio_filename = "temp_tts_output.mp3"  # TTS文件
```

**如何修改**：
```python
# 保存到指定目录
filename = "./temp/temp_voice.wav"
audio_filename = "./temp/temp_tts_output.mp3"

# 使用时间戳命名
import time
filename = f"temp_voice_{int(time.time())}.wav"
audio_filename = f"tts_{int(time.time())}.mp3"
```

### 10.4 录音等待时间配置

**位置**：第1339行
```python
time.sleep(0.3)  # 等待录音线程停止
```

**如何修改**：
```python
time.sleep(0.3)  # 当前值（300毫秒）
# time.sleep(0.5)  # 更长的等待时间（如果录音停止较慢）
# time.sleep(0.1)  # 更短的等待时间（如果录音停止较快）
```

### 10.5 情绪置信度阈值配置

**位置**：第244行和第1386-1389行
```python
if emotion_probability < 0.4:  # 第244行：低置信度时使用neutral
    emotion_mode = "neutral"
    emotion_probability = 0.5
```

**如何修改**：
```python
if emotion_probability < 0.4:  # 当前阈值
# if emotion_probability < 0.5:  # 更严格的阈值
# if emotion_probability < 0.3:  # 更宽松的阈值
```

### 10.6 LLM API超时时间配置

**位置**：第705行
```python
response = requests.post(url, json=data, headers=headers, timeout=15)
```

**如何修改**：
```python
timeout=15  # 当前值（15秒）
# timeout=30  # 更长的超时时间（如果网络较慢）
# timeout=10  # 更短的超时时间（如果网络较快）
```

### 10.7 TTS API超时时间配置

**位置**：第1124行
```python
response = requests.post(url, data=post_data, headers=headers, timeout=10)
```

**如何修改**：
```python
timeout=10  # 当前值（10秒）
# timeout=20  # 更长的超时时间
# timeout=5   # 更短的超时时间
```

### 10.8 录音线程CPU占用控制

**位置**：第299行
```python
time.sleep(0.01)  # 避免CPU占用过高
```

**如何修改**：
```python
time.sleep(0.01)  # 当前值（10毫秒）
# time.sleep(0.02)  # 更长的休眠时间（降低CPU占用）
# time.sleep(0.005)  # 更短的休眠时间（提高响应速度）
```

### 10.9 情绪检测显示文字配置

**位置**：第1326-1329行
```python
cv2.putText(frame, f"Emotion: {emotion_label} ({emotion_conf:.2f})", 
           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(frame, "RECORDING... (Release SPACE to stop)", 
           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
```

**如何修改**：
```python
# 修改显示文字
cv2.putText(frame, f"情绪: {emotion_label} ({emotion_conf:.2f})", 
           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(frame, "录音中... (松开空格键停止)", 
           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# 修改显示位置
cv2.putText(frame, f"Emotion: {emotion_label}", 
           (20, 50), ...)  # 修改(10, 30)为(20, 50)

# 修改字体大小和颜色
cv2.putText(frame, f"Emotion: {emotion_label}", 
           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
# 1.2是字体大小，3是线条粗细，(255, 255, 0)是颜色（BGR格式）
```

---

## 快速参考表

| 配置项 | 位置 | 行号 | 默认值 |
|--------|------|------|--------|
| 摄像头索引 | `init_emotion_module()` | 104 | 1 |
| 录音采样率 | `record_audio()` | 263 | 16000 |
| 情绪平滑窗口 | 全局变量 | 54 | 10 |
| 机器人个性 | 全局变量 | 67-74 | medium/medium/medium/high/low |
| LLM模型 | `call_llm()` | 666 | deepseek-v3-1-terminus |
| LLM温度 | `call_llm()` | 672 | 0.3 |
| TTS音色 | `text_to_speech()` | 1111 | 4193（度泽言） |
| TTS语速 | `text_to_speech()` | 1108 | 5 |
| TTS音调 | `text_to_speech()` | 1109 | 5 |
| TTS音量 | `text_to_speech()` | 1110 | 5 |
| 显示超时（用户） | `main()` | 1376 | 30秒 |
| 显示超时（机器人） | `update_memory()` | 1209 | 10秒 |
| 录音按键 | `on_key_press()` | 1091 | 空格键 |
| 退出按键 | `main()` | 1343 | Q键 |

---

## 常见问题

### Q1: 如何完全禁用TTS功能？
A: 注释掉第1421-1441行的TTS调用代码。

### Q2: 如何修改情绪标签的中文显示？
A: 修改第1056-1067行的`get_emotion_emoji()`函数，添加中文标签映射。

### Q3: 如何添加新的情绪类型？
A: 需要修改情绪模型和标签列表，这是一个较大的改动，需要重新训练模型。

### Q4: 如何保存对话历史？
A: 可以在`update_memory()`函数中添加文件保存逻辑。

### Q5: 如何修改窗口大小？
A: 使用OpenCV的窗口调整功能，或修改第1232行的窗口创建代码。

---

## 技术支持

如有问题，请检查：
1. 所有依赖包是否已安装
2. API密钥是否正确配置
3. 摄像头和麦克风权限是否已授予
4. 网络连接是否正常

---


## 原Emotion程序
### Emotion
This software recognizes human faces and their corresponding emotions from a video or webcam feed. Powered by OpenCV and Deep Learning.

![Demo](https://github.com/petercunha/Emotion/blob/master/demo/demo.gif?raw=true)


#### Installation

Clone the repository:
```
git clone https://github.com/petercunha/Emotion.git
cd Emotion/
```

Install these dependencies with `pip3 install <module name>`
-	tensorflow
-	numpy
-	scipy
-	opencv-python
-	pillow
-	pandas
-	matplotlib
-	h5py
-	keras

Once the dependencies are installed, you can run the project.
`python3 emotions.py`


#### To train new models for emotion classification

- Download the fer2013.tar.gz file from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- Move the downloaded file to the datasets directory inside this repository.
- Untar the file:
`tar -xzf fer2013.tar`
- Download train_emotion_classifier.py from orriaga's repo [here](https://github.com/oarriaga/face_classification/blob/master/src/train_emotion_classifier.py)
- Run the train_emotion_classification.py file:
`python3 train_emotion_classifier.py`


#### Deep Learning Model

The model used is from this [research paper](https://github.com/oarriaga/face_classification/blob/master/report.pdf) written by Octavio Arriaga, Paul G. Plöger, and Matias Valdenegro.

![Model](https://i.imgur.com/vr9yDaF.png?1)


#### Credit

* Computer vision powered by OpenCV.
* Neural network scaffolding powered by Keras with Tensorflow.
* Convolutional Neural Network (CNN) deep learning architecture is from this [research paper](https://github.com/oarriaga/face_classification/blob/master/report.pdf).
* Pretrained Keras model and much of the OpenCV code provided by GitHub user [oarriaga](https://github.com/oarriaga).
