"""
真实交互Demo - 按键录音 + 情绪检测 + LLM生成
按空格键开始录音和情绪检测，松开停止，自动生成回复
"""
import cv2
import numpy as np
import time
import wave
import os
import glob
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("错误：pyaudio未安装")
import base64
import requests
import threading
import urllib.parse
import subprocess
import platform
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
from PIL import Image, ImageDraw, ImageFont

# 尝试导入键盘监听库
try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("警告：pynput未安装，将使用OpenCV的按键检测（可能不够实时）")
    print("安装：pip install pynput")

# ==================== API配置 ====================
# 百度ASR API配置（使用提供的key作为token）
BAIDU_ASR_TOKEN = "bce-v3/ALTAK-UsN0FyDWJeabgG28Nci1z/efb33d53a5e2be537304952a4f3fc1bda41e45d6"  # BCE签名token（备用）
BAIDU_ASR_API_KEY = "oYQuxRUoN93lwd847k782HOF"  # API Key
BAIDU_ASR_SECRET_KEY = "UdbOQqdeWy6tkSso1ItspzMHiboxJX1Q"  # Secret Key
DOUBAO_API_KEY = "0699bcd4-d849-4c1f-a8b5-38847b05531e"

# ==================== 全局变量 ====================
# 情感检测模块
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')
emotion_classifier = None
face_cascade = None
emotion_target_size = None
emotion_offsets = (20, 40)
frame_window = 10
emotion_window = []
cap = None

# 录音相关
is_recording = False
recording_frames = []
audio_stream = None
pyaudio_instance = None

# 记忆模块
short_term_memory = None

# 机器人个性配置（Big Five）
robot_personality = {
    "openness": "medium",          # 开放性：中（不推荐新活动）
    "conscientiousness": "medium",  # 尽责性：中（不提醒）
    "extraversion": "medium",      # 外向性：中（1-2句话）
    "agreeableness": "high",       # 宜人性：高（说共情的话）
    "neuroticism": "low"           # 神经质：低（情绪程度波动小）
}

# 语义记忆（用户偏好、行为模式）
semantic_memory = {
    "user_preferences": {},      # 用户偏好
    "behavior_patterns": [],     # 行为模式
    "emotional_trends": []      # 情绪趋势
}

# 键盘状态
space_pressed = False
listener = None

# 界面显示状态
robot_response_display = None  # 存储最新的机器人回复用于显示
user_voice_display = None  # 存储用户语音转录文字用于显示
display_timeout = 0  # 显示超时时间
user_display_timeout = 0  # 用户语音显示超时时间

# ==================== 初始化函数 ====================

def init_emotion_module():
    """初始化情感检测模块"""
    global emotion_classifier, face_cascade, emotion_target_size, cap
    
    print("初始化情感检测模块...")
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model(emotion_model_path, compile=False)
    emotion_target_size = emotion_classifier.input_shape[1:3]
    
    cap = cv2.VideoCapture(1)  # MacBook前置摄像头
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # 默认摄像头
    
    print("✓ 情感检测模块初始化完成")
    return cap.isOpened()

def get_baidu_access_token(api_key=None, secret_key=None):
    """获取百度ASR Access Token"""
    url = "https://aip.baidubce.com/oauth/2.0/token"
    
    # 如果未提供参数，使用全局变量
    if api_key is None:
        api_key = BAIDU_ASR_API_KEY
    if secret_key is None:
        secret_key = BAIDU_ASR_SECRET_KEY
    
    # 如果全局变量为空，尝试从环境变量读取
    import os
    if not api_key:
        api_key = os.getenv("BAIDU_ASR_API_KEY", "")
    if not secret_key:
        secret_key = os.getenv("BAIDU_ASR_SECRET_KEY", "")
    
    if not api_key or not secret_key:
        return None
    
    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key
    }
    try:
        response = requests.post(url, params=params)
        result = response.json()
        token = result.get("access_token")
        if token:
            print(f"✓ 成功获取access_token")
        return token
    except Exception as e:
        print(f"获取百度ASR Token失败：{e}")
        return None

def init_asr():
    """初始化ASR"""
    global BAIDU_ASR_TOKEN, BAIDU_ASR_API_KEY, BAIDU_ASR_SECRET_KEY
    
    # 优先尝试使用API_KEY和SECRET_KEY获取access_token
    import os
    if BAIDU_ASR_API_KEY == "你的API_KEY" or not BAIDU_ASR_API_KEY:
        BAIDU_ASR_API_KEY = os.getenv("BAIDU_ASR_API_KEY", "")
    if BAIDU_ASR_SECRET_KEY == "你的SECRET_KEY" or not BAIDU_ASR_SECRET_KEY:
        BAIDU_ASR_SECRET_KEY = os.getenv("BAIDU_ASR_SECRET_KEY", "")
    
    # 如果配置了API_KEY和SECRET_KEY，尝试获取access_token
    if BAIDU_ASR_API_KEY and BAIDU_ASR_SECRET_KEY:
        print(f"正在使用API_KEY获取access_token...")
        token = get_baidu_access_token()
        if token:
            print("✓ ASR初始化完成（使用access_token）")
            return {"token": token}
        else:
            print("⚠ 获取access_token失败，将尝试使用BCE签名token")
    
    # 如果获取access_token失败，尝试使用BCE签名token（可能不支持）
    if BAIDU_ASR_TOKEN and BAIDU_ASR_TOKEN != "":
        print(f"⚠ 使用BCE签名token（可能不支持REST API）")
        print("提示：请配置BAIDU_ASR_SECRET_KEY以获取正确的access_token")
        return {"token": BAIDU_ASR_TOKEN}
    
    print("警告：百度ASR API密钥未配置，语音识别功能将不可用")
    print("提示：请配置BAIDU_ASR_SECRET_KEY")
    return None

def init_audio():
    """初始化音频录制"""
    global pyaudio_instance
    if not PYAUDIO_AVAILABLE:
        print("错误：pyaudio未安装，无法录音")
        return False
    try:
        pyaudio_instance = pyaudio.PyAudio()
        print("✓ 音频录制初始化完成")
        return True
    except Exception as e:
        print(f"音频录制初始化失败：{e}")
        return False

# ==================== 核心功能函数 ====================

def detect_emotion_from_frame(frame):
    """从单帧图像检测情绪"""
    global emotion_window
    
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_image, 
        scaleFactor=1.1, 
        minNeighbors=5,
        minSize=(30, 30), 
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return ("neutral", 0.5)
    
    # 处理第一个检测到的人脸
    face_coordinates = faces[0]
    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    gray_face = gray_image[y1:y2, x1:x2]
    
    try:
        gray_face = cv2.resize(gray_face, emotion_target_size)
    except:
        return ("neutral", 0.5)
    
    # 预处理
    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    
    # 预测
    emotion_prediction = emotion_classifier.predict(gray_face, verbose=0)
    emotion_probability = float(np.max(emotion_prediction))
    emotion_label_arg = int(np.argmax(emotion_prediction))
    emotion_text = emotion_labels[emotion_label_arg]
    
    # 确保是7类标签之一
    valid_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    if emotion_text not in valid_labels:
        emotion_text = "neutral"
    
    # 滑动窗口平滑处理
    emotion_window.append(emotion_text)
    if len(emotion_window) > frame_window:
        emotion_window.pop(0)
    
    try:
        emotion_mode = mode(emotion_window)
    except:
        emotion_mode = emotion_text
    
    # 低置信度处理
    if emotion_probability < 0.4:
        emotion_mode = "neutral"
        emotion_probability = 0.5
    
    return (emotion_mode, emotion_probability)

def record_audio():
    """录音线程函数"""
    global is_recording, recording_frames, audio_stream, pyaudio_instance
    
    if not PYAUDIO_AVAILABLE:
        return
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    while True:
        if is_recording and audio_stream is None:
            # 开始录音
            try:
                audio_stream = pyaudio_instance.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK
                )
                recording_frames = []
                print("[录音中...]")
            except Exception as e:
                print(f"录音启动失败：{e}")
        
        elif is_recording and audio_stream:
            # 录音中，收集数据
            try:
                data = audio_stream.read(CHUNK, exception_on_overflow=False)
                recording_frames.append(data)
            except:
                pass
        
        elif not is_recording and audio_stream:
            # 停止录音
            try:
                audio_stream.stop_stream()
                audio_stream.close()
                audio_stream = None
                print("[录音结束]")
            except:
                pass
        
        time.sleep(0.01)  # 避免CPU占用过高

def save_audio_to_file(frames):
    """保存录音到文件"""
    filename = "temp_voice.wav"
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(pyaudio_instance.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

def recognize_speech(audio_file, asr_client):
    """识别语音"""
    if asr_client is None:
        print("ASR客户端未初始化")
        return "未识别到语音"
    
    try:
        # 读取音频文件
        with open(audio_file, 'rb') as f:
            audio_data = f.read()
        
        # 检查音频文件大小
        if len(audio_data) < 1000:  # 小于1KB，可能是空的
            print(f"警告：音频文件太小（{len(audio_data)}字节），可能录音失败")
            return "未识别到语音"
        
        print(f"音频文件大小：{len(audio_data)}字节")
        
        # 使用百度SDK（如果可用且有API_KEY和SECRET_KEY）
        try:
            if BAIDU_ASR_API_KEY and BAIDU_ASR_SECRET_KEY:
                try:
                    from aip import AipSpeech
                    print("使用百度SDK进行识别...")
                    aip_client = AipSpeech("", BAIDU_ASR_API_KEY, BAIDU_ASR_SECRET_KEY)
                    result = aip_client.asr(audio_data, 'wav', 16000, {'dev_pid': 1537})
                    
                    print(f"SDK识别结果：{result}")
                    if result and 'result' in result and len(result['result']) > 0:
                        text = result['result'][0]
                        print(f"✓ 识别成功：{text}")
                        return text
                    elif result and 'err_no' in result:
                        err_no = result.get('err_no')
                        err_msg = result.get('err_msg', '未知错误')
                        print(f"✗ SDK识别错误：err_no={err_no}, err_msg={err_msg}")
                        # 详细错误说明
                        if err_no == 3301:
                            print("  错误说明：token无效或过期")
                        elif err_no == 3302:
                            print("  错误说明：音频格式不支持")
                        elif err_no == 3303:
                            print("  错误说明：音频参数错误")
                        elif err_no == 3304:
                            print("  错误说明：音频质量太差")
                        elif err_no == 3305:
                            print("  错误说明：音频时长太短")
                        elif err_no == 3307:
                            print("  错误说明：识别服务异常")
                        elif err_no == 3308:
                            print("  错误说明：音频时长太长")
                    else:
                        print(f"✗ SDK识别失败：未知错误，结果={result}")
                except ImportError:
                    print("百度SDK未安装，使用REST API...")
                    raise ImportError("使用REST API")
                except Exception as e:
                    print(f"✗ SDK调用失败：{e}")
                    import traceback
                    traceback.print_exc()
                    raise ImportError("使用REST API")
            else:
                raise ImportError("使用REST API")
        except ImportError:
            # 使用REST API - 需要先获取access_token
            print("使用REST API识别...")
            
            # 如果token是bce-v3格式，需要先获取access_token
            token = asr_client.get("token", "")
            if token.startswith("bce-v3/"):
                print("检测到BCE签名token，尝试使用API_KEY获取access_token...")
                # 优先使用代码中配置的API_KEY和SECRET_KEY
                api_key = BAIDU_ASR_API_KEY
                secret_key = BAIDU_ASR_SECRET_KEY
                
                # 如果代码中未配置，尝试从环境变量获取
                if not api_key or not secret_key:
                    import os
                    api_key = os.getenv("BAIDU_ASR_API_KEY", "")
                    secret_key = os.getenv("BAIDU_ASR_SECRET_KEY", "")
                
                if api_key and secret_key:
                    print(f"使用API_KEY获取access_token...")
                    access_token = get_baidu_access_token(api_key, secret_key)
                    if access_token:
                        token = access_token
                        print(f"✓ 成功获取access_token")
                    else:
                        print("✗ 获取access_token失败")
                        return "未识别到语音"
                else:
                    print("⚠ 未配置API_KEY和SECRET_KEY，无法获取access_token")
                    print("提示：请配置BAIDU_ASR_API_KEY和BAIDU_ASR_SECRET_KEY")
                    return "未识别到语音"
            
            # 百度ASR API调用
            # 根据百度ASR文档，短语音识别标准版API端点
            url = "http://vop.baidu.com/server_api"
            
            # 方法1：尝试使用base64编码（推荐，百度ASR REST API推荐方式）
            try:
                # 百度ASR REST API需要使用base64编码的音频数据
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                payload = {
                    "format": "wav",
                    "rate": 16000,
                    "channel": 1,
                    "cuid": "VPMAnb5S3dfr6RPD67qvzmNBO850fDTc",
                    "dev_pid": 1537,  # 1537=中文普通话
                    "token": token,
                    "speech": audio_base64,
                    "len": len(audio_data)
                }
                
                headers = {
                    'Content-Type': 'application/json'
                }
                
                print(f"发送REST API请求（base64编码）...")
                print(f"  URL: {url}")
                print(f"  Token: {token[:20]}...")
                print(f"  音频大小: {len(audio_data)}字节")
                print(f"  格式: wav, 采样率: 16000, 声道: 1")
                print(f"  base64长度: {len(audio_base64)}字符")
                
                response = requests.post(url, json=payload, headers=headers, timeout=10)
                print(f"REST API响应状态码：{response.status_code}")
                print(f"REST API响应内容：{response.text[:500]}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"REST API识别结果：{result}")
                    if result.get('err_no') == 0 and result.get('result'):
                        if len(result['result']) > 0:
                            text = result['result'][0]
                            print(f"✓ 识别成功：{text}")
                            return text
                        else:
                            print("✗ 识别结果为空")
                    elif result.get('err_no'):
                        err_no = result.get('err_no')
                        err_msg = result.get('err_msg', '未知错误')
                        print(f"✗ REST API识别错误：err_no={err_no}, err_msg={err_msg}")
                        # 详细错误说明
                        if err_no == 3301:
                            print("  错误说明：token无效或过期，请检查token")
                        elif err_no == 3302:
                            print("  错误说明：权限不足（No permission to access data）")
                            print("  可能原因：")
                            print("    1. 应用未启用语音识别服务")
                            print("    2. 应用权限配置不正确")
                            print("    3. 应用状态异常")
                            print("  解决方案：")
                            print("    1. 检查应用状态是否为「已启用」")
                            print("    2. 确认应用已启用「语音识别」服务")
                            print("    3. 如果无法修复，重新创建应用")
                        elif err_no == 3303:
                            print("  错误说明：音频参数错误")
                        elif err_no == 3304:
                            print("  错误说明：音频质量太差，请确保录音清晰")
                        elif err_no == 3305:
                            print("  错误说明：音频时长太短（至少0.5秒）")
                        elif err_no == 3307:
                            print("  错误说明：识别服务异常，请稍后重试")
                        elif err_no == 3308:
                            print("  错误说明：音频时长太长（最长60秒）")
                        elif err_no == 3300:
                            print("  错误说明：JSON格式错误，已改用base64编码方式")
                        else:
                            print(f"  错误说明：未知错误码 {err_no}")
                    else:
                        print(f"✗ REST API识别失败：未知错误，结果={result}")
                else:
                    print(f"✗ REST API请求失败：HTTP {response.status_code}")
                    print(f"  响应内容：{response.text[:200]}")
            except Exception as e:
                print(f"REST API调用失败：{e}")
                import traceback
                traceback.print_exc()
                
                # 方法2：如果base64失败，尝试使用multipart/form-data（备用方案）
                try:
                    print("base64方式失败，尝试使用multipart/form-data...")
                    files = {
                        'audio': ('audio.wav', audio_data, 'audio/wav')
                    }
                    data = {
                        'format': 'wav',
                        'rate': '16000',
                        'channel': '1',
                        'cuid': 'VPMAnb5S3dfr6RPD67qvzmNBO850fDTc',
                        'dev_pid': '1537',
                        'token': token
                    }
                    
                    response = requests.post(url, files=files, data=data, timeout=10)
                    print(f"multipart方式响应状态码：{response.status_code}")
                    print(f"multipart方式响应内容：{response.text[:500]}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"multipart方式识别结果：{result}")
                        if result.get('err_no') == 0 and result.get('result'):
                            if len(result['result']) > 0:
                                text = result['result'][0]
                                print(f"✓ 识别成功（multipart）：{text}")
                                return text
                        elif result.get('err_no'):
                            err_no = result.get('err_no')
                            err_msg = result.get('err_msg', '未知错误')
                            print(f"✗ multipart方式错误：err_no={err_no}, err_msg={err_msg}")
                except Exception as e2:
                    print(f"multipart方式也失败：{e2}")
                    import traceback
                    traceback.print_exc()
        
        return "未识别到语音"
    except Exception as e:
        print(f"语音识别失败：{e}")
        import traceback
        traceback.print_exc()
        return "未识别到语音"

def init_robot_personality():
    """初始化机器人个性配置"""
    global robot_personality
    print("\n" + "=" * 60)
    print("【机器人个性配置初始化】")
    print("=" * 60)
    print(f"开放性（Openness）：{robot_personality['openness']} - 不推荐新活动")
    print(f"尽责性（Conscientiousness）：{robot_personality['conscientiousness']} - 不提醒")
    print(f"外向性（Extraversion）：{robot_personality['extraversion']} - 1-2句话")
    print(f"宜人性（Agreeableness）：{robot_personality['agreeableness']} - 说共情的话")
    print(f"神经质（Neuroticism）：{robot_personality['neuroticism']} - 情绪程度波动小")
    print("=" * 60)
    return robot_personality

def extract_semantic_memory(memory):
    """提取语义记忆（用户偏好、行为模式）"""
    global semantic_memory
    
    if memory is None:
        return {}
    
    # 提取用户偏好（从交互历史中）
    preferences = {}
    if 'user_voice' in memory:
        voice = memory['user_voice']
        # 简单的偏好提取（可以根据实际需求扩展）
        if '谢谢' in voice or '感谢' in voice:
            preferences['appreciates_gratitude'] = True
        if '幽默' in voice or '搞笑' in voice:
            preferences['likes_humor'] = True
        if '名字' in voice or '叫什么' in voice:
            preferences['curious_about_identity'] = True
    
    # 更新语义记忆
    semantic_memory['user_preferences'].update(preferences)
    
    return {
        'preferences': preferences,
        'all_preferences': semantic_memory['user_preferences']
    }

def get_memory_context(memory):
    """生成记忆上下文用于prompt"""
    if memory is None:
        return "无历史交互"
    
    # 基本记忆上下文
    memory_text = f"上轮用户说：{memory.get('user_voice', '')}，情绪是{memory.get('user_emotion', 'neutral')}；机器人回复：{memory.get('robot_response', '')}，情绪是{memory.get('robot_emotion', 'neutral')}（程度{memory.get('robot_emotion_level', 0.7)}）"
    
    # 添加语义记忆
    semantic = extract_semantic_memory(memory)
    if semantic.get('preferences'):
        prefs = []
        if semantic['preferences'].get('likes_humor'):
            prefs.append("用户喜欢幽默")
        if semantic['preferences'].get('appreciates_gratitude'):
            prefs.append("用户表达感谢")
        if prefs:
            memory_text += f"；用户偏好：{', '.join(prefs)}"
    
    return memory_text

def call_llm(user_sync_data, memory):
    """调用豆包LLM生成回复"""
    global robot_personality
    
    # 生成记忆上下文
    memory_context = get_memory_context(memory)
    
    # 构造提示词（基于论文的五大人格理论和Appraisal理论）
    system_prompt = f"""你是"温和共情型"机器人。请严格按照以下规则工作，最后只输出3行结果。

【个性参数化（Big Five）】
你的个性参数：
- 开放性（Openness）：{robot_personality['openness']} - 不推荐新活动
- 尽责性（Conscientiousness）：{robot_personality['conscientiousness']} - 不提醒
- 外向性（Extraversion）：{robot_personality['extraversion']} - 1-2句话
- 宜人性（Agreeableness）：{robot_personality['agreeableness']} - 说共情的话
- 神经质（Neuroticism）：{robot_personality['neuroticism']} - 情绪程度波动小

【Appraisal评估步骤】
请按照以下步骤评估事件：
1. 相关性检查：事件是否与机器人目标相关？（用户的情感状态）
2. 效价评估：事件的情感效价（positive/negative/sarcasm/neutral）
   - 注意：用户语音的字面意思和语气可能不一致（如反讽）
3. 应对潜力：机器人能否缓解负面情绪？如何回应？

【行为生成规则】
基于评估结果和个性参数生成：
- 情感状态：根据Appraisal结果和用户情绪生成你的情感
  - 用户情绪置信度≥0.6：按用户情绪生成你的情感（共情）
  - 用户情绪置信度<0.6：默认neutral:0.7
- 行动选择：根据个性参数和记忆上下文选择回应方式
  - 宜人性高：说共情的话
  - 外向性中：1-2句话
  - 考虑用户偏好（如喜欢幽默）

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
    
    user_prompt = f"""当前用户数据：
- 语音：{user_sync_data['voice_text']}
- 情绪：{user_sync_data['emotion_label']}（置信度：{user_sync_data['emotion_conf']:.2f}）

记忆上下文：
{memory_context}

任务：
1. 使用Appraisal理论评估事件（相关性、效价、应对潜力）
2. 根据个性参数生成情感状态
3. 选择行动（考虑记忆上下文中的用户偏好）
4. 输出3行结果（情绪标签、情绪程度、回复文字）"""
    
    # 调用API
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    headers = {
        "Authorization": f"Bearer {DOUBAO_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-v3-1-terminus",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_completion_tokens": 100,  # 增加token数，让LLM有足够空间输出3行
        "temperature": 0.3  # 适中的temperature，保持创造力但不过度推理
    }
    
    try:
        print(f"\n" + "=" * 60)
        print(f"【调用豆包LLM API】")
        print("=" * 60)
        
        # 显示机器人个性配置
        print("\n【机器人个性配置】")
        print(f"  开放性（Openness）：{robot_personality['openness']}")
        print(f"  尽责性（Conscientiousness）：{robot_personality['conscientiousness']}")
        print(f"  外向性（Extraversion）：{robot_personality['extraversion']}")
        print(f"  宜人性（Agreeableness）：{robot_personality['agreeableness']}")
        print(f"  神经质（Neuroticism）：{robot_personality['neuroticism']}")
        
        # 显示用户数据
        print("\n【用户数据】")
        print(f"  用户语音: {user_sync_data['voice_text']}")
        print(f"  用户情绪: {user_sync_data['emotion_label']} (置信度: {user_sync_data['emotion_conf']:.2f})")
        
        # 显示记忆上下文
        print("\n【记忆上下文】")
        print(f"  {memory_context}")
        
        # 显示Appraisal评估提示
        print("\n【Appraisal评估步骤】")
        print("  1. 相关性检查：事件是否与机器人目标相关？")
        print("  2. 效价评估：事件的情感效价（positive/negative/sarcasm/neutral）")
        print("  3. 应对潜力：机器人能否缓解负面情绪？")
        
        print("\n机器人思考中...")
        
        response = requests.post(url, json=data, headers=headers, timeout=15)
        response.raise_for_status()
        result = response.json()
        
        result_content = str(result)
        if len(result_content) > 500:
            result_content = result_content[:500] + "..."
        print(f"\n【LLM API响应】")
        print(f"  {result_content}")
        
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]
            llm_text = ""
            
            # 优先使用content字段（如果有内容）
            if "content" in message and message["content"] and message["content"].strip():
                llm_text = message["content"].strip()
                print(f"LLM输出文本（从content）: {llm_text}")
            elif "reasoning_content" in message and message["reasoning_content"]:
                # 如果只有reasoning_content，说明LLM输出了推理过程
                # 尝试从reasoning_content中提取真正的3行输出
                reasoning_text = message["reasoning_content"].strip()
                print(f"\n【LLM推理过程】")
                print("=" * 60)
                print(reasoning_text)
                print("=" * 60)
                print(f"\n⚠ 从推理过程中提取3行输出...")
                
                # 尝试从reasoning_content中提取真正的3行输出
                # 方法1：查找最后几行，看是否有完整的3行输出
                lines = [line.strip() for line in reasoning_text.split("\n") if line.strip()]
                valid_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
                
                # 从后往前查找，找到包含情绪标签的行
                output_lines = []
                found_label = False
                
                # 先查找最后10行，看是否有完整的3行输出
                for i in range(len(lines) - 1, max(-1, len(lines) - 15), -1):
                    line_lower = lines[i].lower().strip()
                    # 检查是否是纯情绪标签
                    for label in valid_labels:
                        if label == line_lower:
                            # 找到情绪标签行，检查后续是否有2行
                            if i + 2 < len(lines):
                                # 检查后续行是否符合格式（第2行是数字，第3行是回复）
                                try:
                                    float(lines[i + 1].strip())
                                    # 找到了完整的3行输出
                                    output_lines = lines[i:i+3]
                                    found_label = True
                                    print(f"✓ 从推理过程中找到完整3行输出（第{i}-{i+2}行）: {output_lines}")
                                    break
                                except:
                                    pass
                    if found_label:
                        break
                
                # 方法2：如果没找到，尝试从推理内容中提取LLM想出的回复
                if not output_lines:
                    # 查找推理过程中提到的回复文字（通常在引号内或"比如"后面）
                    import re
                    # 查找引号内的内容（可能是回复文字）
                    # 匹配中文引号、英文引号、单引号等
                    quotes = re.findall(r'["""]([^""""]+)["""]', reasoning_text)
                    if not quotes:
                        # 如果没有找到引号，尝试查找"比如"、"例如"后面的内容
                        examples = re.findall(r'(?:比如|例如|如|像是|像)[：:]?\s*["""]?([^？。！？\n]{5,30})["""]?', reasoning_text)
                        if examples:
                            quotes = examples
                    
                    if quotes:
                        # 找到最后一个引号内的内容（通常是最终答案）
                        potential_response = quotes[-1].strip()
                        # 从推理过程中提取情绪标签和程度
                        emotion_from_reasoning = None
                        level_from_reasoning = 0.7
                        
                        # 查找情绪标签（更精确的匹配）
                        for label in valid_labels:
                            # 查找"情绪标签是xxx"、"标签是xxx"、"是xxx"等
                            patterns = [
                                f"情绪标签是{label}",
                                f"标签是{label}",
                                f"情绪标签{label}",
                                f"标签{label}",
                                f"所以情绪标签是{label}",
                                f"所以标签是{label}",
                            ]
                            for pattern in patterns:
                                if pattern in reasoning_text:
                                    emotion_from_reasoning = label
                                    print(f"✓ 从推理过程中提取情绪标签: {label}")
                                    break
                            if emotion_from_reasoning:
                                break
                        
                        # 如果没有找到，使用规则判断（作为备选方案）
                        if not emotion_from_reasoning:
                            user_conf = user_sync_data['emotion_conf']
                            if user_conf >= 0.6:
                                emotion_from_reasoning = user_sync_data['emotion_label']
                            else:
                                emotion_from_reasoning = "neutral"
                            print(f"⚠ 无法从推理过程中提取情绪标签，使用规则判断: {emotion_from_reasoning}")
                        
                        # 查找情绪程度（更精确的匹配）
                        level_matches = re.findall(r'程度\s*([0-9.]+)', reasoning_text)
                        if not level_matches:
                            # 尝试查找"0.7"、"0.8"等数字
                            level_matches = re.findall(r'\b([0-9]\.[0-9])\b', reasoning_text)
                        
                        if level_matches:
                            try:
                                # 使用最后一个匹配的数字（通常是最终答案）
                                level_from_reasoning = float(level_matches[-1])
                                print(f"✓ 从推理过程中提取情绪程度: {level_from_reasoning}")
                            except:
                                pass
                        else:
                            # 如果没有找到，使用默认值
                            level_from_reasoning = 0.7
                        
                        # 如果找到了回复文字，构造3行输出
                        if potential_response and len(potential_response) > 5:
                            output_lines = [emotion_from_reasoning, str(level_from_reasoning), potential_response]
                            print(f"✓ 从推理过程中提取回复文字: {potential_response}")
                            print(f"✓ 构造3行输出: {output_lines}")
                
                # 如果找到了输出行，使用它们
                if output_lines and len(output_lines) >= 3:
                    llm_text = "\n".join(output_lines)
                    print(f"✓ 使用从推理过程中提取的3行: {llm_text}")
                else:
                    # 如果没找到，尝试使用LLM的完整推理内容，让后续解析逻辑处理
                    print(f"⚠ 无法从推理过程中提取3行，使用完整推理内容")
                    llm_text = reasoning_text
            
            if not llm_text:
                print(f"✗ LLM输出为空: {message}")
                raise Exception("LLM输出为空")
        else:
            print(f"✗ API返回格式错误: {result}")
            raise Exception("API返回格式错误")
        
        # 解析为3类结果
        # 如果LLM输出了推理过程，需要提取真正的3行输出
        lines_raw = llm_text.split("\n")
        lines = [line.strip() for line in lines_raw if line.strip()]
        
        print(f"LLM输出原始行数: {len(lines_raw)}")
        print(f"LLM输出非空行数: {len(lines)}")
        
        # 如果输出包含推理过程，需要提取真正的3行输出
        valid_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        
        # 方法1：查找纯情绪标签行（单独一行，只包含情绪标签）
        output_lines = []
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            # 检查是否是纯情绪标签（不包含其他文字）
            for label in valid_labels:
                if label == line_lower:
                    # 找到纯情绪标签行，提取后续3行
                    if i + 2 < len(lines):
                        output_lines = lines[i:i+3]
                        print(f"✓ 找到纯情绪标签行（第{i}行），提取3行: {output_lines}")
                        break
            if output_lines:
                break
        
        # 方法2：如果方法1失败，从后往前查找包含情绪标签的行
        if not output_lines:
            for i in range(len(lines) - 1, -1, -1):
                line_lower = lines[i].lower().strip()
                for label in valid_labels:
                    # 检查是否是情绪标签行（可能是纯标签或包含少量文字）
                    if label == line_lower or (label in line_lower and len(line_lower) < 15):
                        # 找到情绪标签行，提取后续3行
                        if i + 2 < len(lines):
                            output_lines = lines[i:i+3]
                            print(f"✓ 找到情绪标签行（第{i}行），提取3行: {output_lines}")
                            break
                if output_lines:
                    break
        
        # 方法3：如果还是找不到，使用最后3行（如果包含情绪标签）
        if not output_lines and len(lines) >= 3:
            last_lines = lines[-3:]
            has_label = False
            for line in last_lines:
                for label in valid_labels:
                    if label in line.lower():
                        has_label = True
                        break
                if has_label:
                    break
            if has_label:
                output_lines = last_lines
                print(f"✓ 使用最后3行（包含情绪标签）: {output_lines}")
        
        # 如果找到了输出行，使用它们
        if output_lines and len(output_lines) >= 3:
            lines = output_lines
            print(f"✓ 最终提取的3行: {lines}")
        elif len(lines) > 3:
            # 如果超过3行但没找到情绪标签，使用最后3行
            lines = lines[-3:]
            print(f"⚠ 使用最后3行: {lines}")
        
        if len(lines) < 3:
            # 容错处理
            print(f"⚠ LLM输出格式不完整，尝试解析...")
            print(f"  LLM输出: {llm_text[:200]}...")
            print(f"  行数: {len(lines)}")
            emotion_label = "neutral"
            emotion_level = 0.7
            response_text = llm_text if llm_text else "我有点没听清，能再说说吗？"
            
            valid_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
            for label in valid_labels:
                if label in lines[0].lower() if len(lines) > 0 else "":
                    emotion_label = label
                    break
            
            import re
            if len(lines) > 1:
                numbers = re.findall(r'0\.\d+|1\.0', lines[1])
                if numbers:
                    emotion_level = float(numbers[0])
            
            if len(lines) >= 3:
                response_text = lines[2]
            elif len(lines) >= 2:
                response_text = lines[1]
        else:
            # 提取情绪标签（只取第一个有效标签）
            emotion_label = lines[0].strip()
            # 清理情绪标签，只保留标签本身
            valid_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
            for label in valid_labels:
                if label in emotion_label.lower():
                    emotion_label = label
                    break
            
            # 提取情绪程度（只取数字）
            try:
                import re
                numbers = re.findall(r'0\.\d+|1\.0', lines[1].strip())
                if numbers:
                    emotion_level = float(numbers[0])
                else:
                    emotion_level = 0.7
            except:
                emotion_level = 0.7
            
            # 提取回复文字（只取回复内容，去掉推理过程）
            response_text = lines[2].strip()
            # 如果回复文字包含推理过程，尝试提取真正的回复
            # 查找引号内的内容或直接提取前几句话
            if len(response_text) > 100:  # 如果太长，可能是推理过程
                # 尝试提取引号内的内容
                import re
                quotes = re.findall(r'[""](.*?)[""]', response_text)
                if quotes:
                    response_text = quotes[0]
                else:
                    # 如果没有引号，提取前50个字符
                    response_text = response_text[:50]
                    print(f"⚠ 回复文字过长，截取前50字符: {response_text}")
            
            print(f"✓ 解析结果: 情绪={emotion_label}, 程度={emotion_level}, 回复={response_text[:50]}...")
        
        # 验证情绪标签（只验证格式，不强制调整）
        valid_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        if emotion_label not in valid_labels:
            print(f"⚠ LLM输出的情绪标签无效: {emotion_label}，使用neutral")
            emotion_label = "neutral"
        
        # 显示LLM原始输出（在调整前）
        print(f"\n【LLM原始输出】")
        print("=" * 60)
        print(f"情绪标签：{emotion_label}")
        print(f"情绪程度：{emotion_level}")
        print(f"回复文字：{response_text[:50]}...")
        print("=" * 60)
        
        # 注意：不再强制覆盖LLM的输出
        # LLM已经在prompt中被告知了Appraisal规则，应该按照规则生成
        # 如果LLM输出不符合规则，会在终端显示警告，但仍然使用LLM的输出
        
        # 显示最终输出结果
        print(f"\n【最终输出结果】（使用LLM生成的值）")
        print("=" * 60)
        print(f"情绪标签：{emotion_label}")
        print(f"情绪程度：{emotion_level}")
        print(f"回复文字：{response_text}")
        print("=" * 60)
        
        return {
            "emotion_label": emotion_label,
            "emotion_level": emotion_level,
            "response_text": response_text
        }
    except Exception as e:
        print(f"✗ LLM调用失败：{e}")
        import traceback
        traceback.print_exc()
        print("⚠ 使用默认回复")
        return {
            "emotion_label": "neutral",
            "emotion_level": 0.7,
            "response_text": "我有点没听清，能再说说吗？"
        }

def cleanup_temp_files():
    """清理临时文件和缓存文件以保护隐私"""
    try:
        # 清理录音文件和TTS音频文件
        audio_files = glob.glob("temp_voice.wav") + glob.glob("temp_tts_output.mp3") + glob.glob("*.wav") + glob.glob("*.mp3")
        for file in audio_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"✓ 已删除临时文件: {file}")
            except Exception as e:
                print(f"⚠ 删除文件失败 {file}: {e}")
        
        # 清理Python缓存文件
        cache_dirs = glob.glob("__pycache__") + glob.glob("**/__pycache__", recursive=True)
        for cache_dir in cache_dirs:
            try:
                if os.path.isdir(cache_dir):
                    import shutil
                    shutil.rmtree(cache_dir)
                    print(f"✓ 已删除缓存目录: {cache_dir}")
            except Exception as e:
                print(f"⚠ 删除缓存目录失败 {cache_dir}: {e}")
        
        # 清理其他可能的临时文件
        temp_files = glob.glob("*.tmp") + glob.glob("*.cache")
        for file in temp_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"✓ 已删除临时文件: {file}")
            except Exception as e:
                print(f"⚠ 删除文件失败 {file}: {e}")
    except Exception as e:
        print(f"⚠ 清理临时文件时出错: {e}")

def get_emotion_emoji(emotion_label):
    """根据情绪标签返回颜文字表情"""
    emoji_map = {
        "angry": "(╯°□°）╯",      # 愤怒
        "disgust": "(￣へ￣)",     # 厌恶
        "fear": "(>_<)",          # 恐惧
        "happy": "(◕‿◕)",         # 开心
        "sad": "(╥_╥)",           # 悲伤
        "surprise": "(⊙_⊙)",      # 惊讶
        "neutral": "(・_・)"       # 中性
    }
    return emoji_map.get(emotion_label, "(・_・)")

def text_to_speech(text, access_token=None):
    """使用百度TTS API将文本转换为语音
    
    Args:
        text: 要合成的文本
        access_token: 百度API的access_token，如果为None则自动获取
    
    Returns:
        音频文件路径，如果失败返回None
    """
    try:
        # 获取access_token
        if access_token is None:
            access_token = get_baidu_access_token()
            if not access_token:
                print("✗ TTS失败：无法获取access_token")
                return None
        
        # 百度TTS API地址
        url = "https://tsn.baidu.com/text2audio"
        
        # 参数准备
        # tex需要2次urlencode（根据百度文档要求）
        # 注意：不能使用requests的data参数自动编码，因为需要tex单独2次编码
        tex_encoded = urllib.parse.quote(text, safe='')
        tex_encoded = urllib.parse.quote(tex_encoded, safe='')
        
        # 用户唯一标识（使用UUID）
        import uuid
        cuid = str(uuid.uuid4())[:60]
        
        # 手动构建POST请求体（表单格式）
        # tex使用2次编码，其他参数直接使用（access_token和cuid通常不包含特殊字符）
        params_list = [
            f"tex={tex_encoded}",  # tex已经2次编码
            f"tok={access_token}",  # access_token通常不需要编码
            f"cuid={cuid}",  # cuid通常不需要编码
            "ctp=1",
            "lan=zh",
            "spd=5",
            "pit=5",
            "vol=5",
            "per=4193",  # 度泽言（大模型音库）
            "aue=3"
        ]
        post_data = "&".join(params_list)
        
        print(f"\n【调用百度TTS API】")
        print(f"  原始文本：{text[:50]}...")
        print(f"  音色：度泽言（4193）")
        
        # 发送POST请求（手动构建的请求体）
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        response = requests.post(url, data=post_data, headers=headers, timeout=10)
        
        # 检查响应
        content_type = response.headers.get('Content-Type', '')
        
        if content_type.startswith('audio'):
            # 合成成功，保存音频文件
            audio_filename = "temp_tts_output.mp3"
            with open(audio_filename, 'wb') as f:
                f.write(response.content)
            
            file_size = len(response.content)
            print(f"✓ TTS成功：已生成音频文件 {audio_filename}（{file_size}字节）")
            return audio_filename
        else:
            # 合成失败，返回错误信息
            try:
                error_info = response.json()
                err_no = error_info.get('err_no', '未知')
                err_msg = error_info.get('err_msg', '未知错误')
                print(f"✗ TTS失败：err_no={err_no}, err_msg={err_msg}")
            except:
                print(f"✗ TTS失败：HTTP {response.status_code}, {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"✗ TTS调用异常：{e}")
        import traceback
        traceback.print_exc()
        return None

def play_audio(audio_file):
    """播放音频文件
    
    Args:
        audio_file: 音频文件路径
    """
    if not audio_file or not os.path.exists(audio_file):
        print(f"✗ 音频文件不存在：{audio_file}")
        return False
    
    try:
        system = platform.system()
        
        if system == "Darwin":  # macOS
            # 使用afplay命令播放
            subprocess.Popen(['afplay', audio_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✓ 正在播放音频：{audio_file}")
            return True
        elif system == "Linux":
            # 尝试使用aplay或mpg123
            try:
                subprocess.Popen(['mpg123', '-q', audio_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"✓ 正在播放音频：{audio_file}")
                return True
            except:
                try:
                    subprocess.Popen(['aplay', audio_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"✓ 正在播放音频：{audio_file}")
                    return True
                except:
                    print("⚠ 无法播放音频：未找到播放器（请安装mpg123或aplay）")
                    return False
        elif system == "Windows":
            # 使用Windows的start命令
            subprocess.Popen(['start', audio_file], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✓ 正在播放音频：{audio_file}")
            return True
        else:
            print(f"⚠ 不支持的操作系统：{system}")
            return False
    except Exception as e:
        print(f"✗ 播放音频失败：{e}")
        return False

def update_memory(user_sync_data, robot_output):
    """更新记忆"""
    global short_term_memory, robot_response_display, display_timeout, semantic_memory
    
    short_term_memory = {
        "user_voice": user_sync_data["voice_text"],
        "user_emotion": user_sync_data["emotion_label"],
        "robot_emotion": robot_output["emotion_label"],
        "robot_emotion_level": robot_output["emotion_level"],
        "robot_response": robot_output["response_text"]
    }
    
    # 更新语义记忆（提取用户偏好）
    extract_semantic_memory(short_term_memory)
    
    # 更新界面显示
    robot_response_display = robot_output
    display_timeout = time.time() + 10  # 显示10秒

# ==================== 主程序 ====================

def on_key_press(key):
    """按键按下回调"""
    global space_pressed
    try:
        if key == keyboard.Key.space:
            space_pressed = True
    except:
        pass

def on_key_release(key):
    """按键释放回调"""
    global space_pressed
    try:
        if key == keyboard.Key.space:
            space_pressed = False
        elif key == keyboard.Key.esc:
            return False  # 停止监听
    except:
        pass

def main():
    """主程序"""
    global is_recording, recording_frames, cap, user_voice_display, user_display_timeout, space_pressed, listener
    
    print("=" * 60)
    print("机器人情感交互系统 - 真实交互Demo")
    print("=" * 60)
    print()
    print("操作说明：")
    print("  1. 按【空格键】开始录音和情绪检测")
    print("  2. 松开【空格键】停止录音")
    print("  3. 系统自动识别语音和情绪，生成回复")
    print("  4. 按【ESC键】退出程序")
    print()
    print("=" * 60)
    print()
    
    # 初始化模块
    if not init_emotion_module():
        print("错误：摄像头初始化失败")
        return
    
    asr_client = init_asr()
    if not init_audio():
        print("错误：音频录制初始化失败")
        return
    
    # 启动录音线程
    audio_thread = threading.Thread(target=record_audio, daemon=True)
    audio_thread.start()
    
    # 启动键盘监听（如果可用）
    if KEYBOARD_AVAILABLE:
        listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
        listener.start()
        print("✓ 键盘监听已启动（使用pynput）")
    else:
        print("⚠ 使用OpenCV按键检测（可能不够实时）")
    
    # 等待一下让线程启动
    time.sleep(0.5)
    
    # 初始化机器人个性配置
    init_robot_personality()
    
    print("\n系统就绪！按空格键开始交互...")
    print()
    
    # 主循环
    cv2.namedWindow('情感交互系统', cv2.WINDOW_NORMAL)
    
    last_key_state = False
    emotions_during_recording = []
    should_exit = False
    
    while not should_exit:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测按键状态
        if KEYBOARD_AVAILABLE:
            current_key_state = space_pressed
        else:
            # 使用OpenCV的waitKey作为备用方案
            key = cv2.waitKey(1) & 0xFF
            current_key_state = (key == ord(' ') or key == 32)
            if key == ord('q') or key == ord('Q'):
                should_exit = True
        
        # 检查空格键状态
        if current_key_state:
            if not last_key_state:
                # 刚按下空格键，开始录音和情绪检测
                is_recording = True
                emotions_during_recording = []
                print("\n" + "=" * 60)
                print("【开始录音和情绪检测】请说话...")
                print("=" * 60)
            last_key_state = True
            
            # 检测情绪
            emotion_label, emotion_conf = detect_emotion_from_frame(frame)
            emotions_during_recording.append((emotion_label, emotion_conf))
            
            # 显示当前情绪
            cv2.putText(frame, f"Emotion: {emotion_label} ({emotion_conf:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "RECORDING... (Release SPACE to stop)", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        elif last_key_state:
            # 刚松开空格键，停止录音并处理
            is_recording = False
            last_key_state = False
            
            print("\n【停止录音】正在处理...")
            
            # 等待录音线程停止
            time.sleep(0.3)
            
            # 计算最终情绪（使用mode）
            if emotions_during_recording:
                emotion_list = [emo for emo, conf in emotions_during_recording]
                try:
                    final_emotion = mode(emotion_list)
                except:
                    final_emotion = emotions_during_recording[-1][0]
                
                # 计算该情绪的平均置信度
                confs = [conf for emo, conf in emotions_during_recording if emo == final_emotion]
                final_conf = sum(confs) / len(confs) if confs else 0.5
            else:
                final_emotion = "neutral"
                final_conf = 0.5
            
            # 保存录音文件
            if recording_frames:
                print(f"录音数据帧数：{len(recording_frames)}")
                total_bytes = sum(len(f) for f in recording_frames)
                print(f"录音总字节数：{total_bytes}字节")
                
                audio_file = save_audio_to_file(recording_frames)
                
                # 检查文件是否保存成功
                import os
                if os.path.exists(audio_file):
                    file_size = os.path.getsize(audio_file)
                    print(f"音频文件已保存：{audio_file}，大小：{file_size}字节")
                    
                    # 识别语音
                    print("正在识别语音...")
                    voice_text = recognize_speech(audio_file, asr_client)
                    
                    # 更新界面显示用户语音（无论识别成功与否都显示）
                    user_voice_display = voice_text
                    user_display_timeout = time.time() + 30  # 显示30秒，让用户有足够时间看到
                    print(f"✓ 用户语音已设置显示: {voice_text}")
                else:
                    print(f"错误：音频文件保存失败")
                    voice_text = "未识别到语音"
                    # 更新界面显示用户语音
                    user_voice_display = voice_text
                    user_display_timeout = time.time() + 30  # 显示30秒
                    print(f"✓ 用户语音已设置显示: {voice_text}")
                
                if voice_text == "未识别到语音":
                    print("警告：语音识别失败，使用默认情绪")
                    final_emotion = "neutral"
                    final_conf = 0.5
                else:
                    print(f"✓ 识别成功：{voice_text}")
                
                print(f"检测情绪：{final_emotion}（置信度：{final_conf:.2f}）")
                
                # 同步数据
                user_sync_data = {
                    "voice_text": voice_text,
                    "emotion_label": final_emotion,
                    "emotion_conf": final_conf
                }
                
                # 调用LLM
                print("机器人思考中...")
                robot_output = call_llm(user_sync_data, short_term_memory)
                
                # 显示结果（控制台）
                print("\n" + "=" * 60)
                print("【机器人回复】")
                print("=" * 60)
                emoji = get_emotion_emoji(robot_output['emotion_label'])
                print(f"{emoji} 情绪标签：{robot_output['emotion_label']}")
                print(f"   情绪程度：{robot_output['emotion_level']}")
                print(f"   回复文字：{robot_output['response_text']}")
                print("=" * 60)
                print("（界面中也会显示回复，持续10秒）")
                print()
                
                # 更新记忆（同时更新界面显示）
                update_memory(user_sync_data, robot_output)
                
                # 将回复文字转为语音并播放
                print("\n【文字转语音】")
                tts_audio_file = text_to_speech(robot_output['response_text'])
                if tts_audio_file:
                    # 在后台线程中播放音频，避免阻塞主程序
                    def play_tts_audio():
                        play_audio(tts_audio_file)
                        # 等待一段时间后删除临时文件（给播放留出时间）
                        # 假设音频最长30秒，等待35秒后删除
                        time.sleep(35)
                        try:
                            if os.path.exists(tts_audio_file):
                                os.remove(tts_audio_file)
                                print(f"✓ 已清理TTS临时文件：{tts_audio_file}")
                        except:
                            pass
                    
                    audio_thread = threading.Thread(target=play_tts_audio, daemon=True)
                    audio_thread.start()
                else:
                    print("⚠ TTS失败，跳过语音播放")
                
                # 清空录音帧
                recording_frames = []
            else:
                print("未检测到录音数据")
            
            emotions_during_recording = []
        
        # 在界面上显示信息（使用PIL绘制中文和颜文字）
        # 用户语音和AI回复同时显示在一个界面中
        show_user_voice = user_voice_display and time.time() < user_display_timeout
        show_robot_response = robot_response_display and time.time() < display_timeout
        
        if show_user_voice or show_robot_response:
            
            # 转换为PIL Image以支持中文和颜文字
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            
            # 尝试加载中文字体
            try:
                font_path = "/System/Library/Fonts/PingFang.ttc"
                font_large = ImageFont.truetype(font_path, 24)
                font_small = ImageFont.truetype(font_path, 20)
                font_medium = ImageFont.truetype(font_path, 22)
            except:
                try:
                    font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
                    font_large = ImageFont.truetype(font_path, 24)
                    font_small = ImageFont.truetype(font_path, 20)
                    font_medium = ImageFont.truetype(font_path, 22)
                except:
                    font_large = ImageFont.load_default()
                    font_small = ImageFont.load_default()
                    font_medium = ImageFont.load_default()
            
            # 计算显示区域位置
            frame_height = frame.shape[0]
            frame_width = frame.shape[1]
            
            # 用户语音显示区域（底部，固定高度60像素）
            user_y_start = frame_height - 60
            user_y_end = frame_height - 10
            
            # 机器人回复显示区域（用户语音上方，根据是否有用户语音调整位置）
            if show_user_voice:
                robot_y_start = frame_height - 160
                robot_y_end = frame_height - 70
            else:
                robot_y_start = frame_height - 100
                robot_y_end = frame_height - 10
            
            # 先绘制背景（半透明）
            overlay = frame.copy()
            
            # 绘制用户语音背景（如果显示）
            if show_user_voice:
                # 根据识别结果使用不同背景色
                if user_voice_display == "未识别到语音":
                    bg_color = (60, 30, 30)  # 红色背景表示识别失败
                else:
                    bg_color = (30, 60, 30)  # 绿色背景表示识别成功
                
                cv2.rectangle(overlay, (10, user_y_start), (frame_width - 10, user_y_end), bg_color, -1)
            
            # 绘制机器人回复背景（如果显示）
            if show_robot_response:
                cv2.rectangle(overlay, (10, robot_y_start), (frame_width - 10, robot_y_end), (30, 30, 60), -1)
            
            # 应用半透明效果
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # 重新转换为PIL Image（因为overlay改变了frame）
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            
            # 重新加载字体
            try:
                font_path = "/System/Library/Fonts/PingFang.ttc"
                font_medium = ImageFont.truetype(font_path, 22)
                font_small = ImageFont.truetype(font_path, 20)
            except:
                try:
                    font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
                    font_medium = ImageFont.truetype(font_path, 22)
                    font_small = ImageFont.truetype(font_path, 20)
                except:
                    font_medium = ImageFont.load_default()
                    font_small = ImageFont.load_default()
            
            # 显示用户语音（底部）
            if show_user_voice:
                if user_voice_display == "未识别到语音":
                    user_text = "你说：（未识别到语音）"
                    text_color = (255, 150, 150)  # 浅红色文字
                else:
                    user_text = f"你说：{user_voice_display}"
                    text_color = (200, 255, 200)  # 浅绿色文字
                
                # 如果文字太长，截断
                if len(user_text) > 50:
                    user_text = user_text[:47] + "..."
                
                draw.text((20, user_y_start + 15), user_text, fill=text_color, font=font_medium)
            
            # 显示机器人回复（用户语音上方）
            if show_robot_response:
                emotion_label = robot_response_display.get('emotion_label', 'neutral')
                emotion_level = robot_response_display.get('emotion_level', 0.7)
                response_text = robot_response_display.get('response_text', '')
                
                # 获取颜文字表情
                emoji = get_emotion_emoji(emotion_label)
                
                # 显示情绪标签和程度
                emotion_text = f"{emoji} 情绪：{emotion_label}（程度：{emotion_level}）"
                draw.text((20, robot_y_start + 10), emotion_text, fill=(255, 255, 255), font=font_medium)
                
                # 显示回复文字（自动换行）
                max_width = frame_width - 40
                line_height = 22
                
                # 文本换行处理
                response_lines = []
                current_line = ""
                for char in response_text:
                    test_line = current_line + char
                    bbox = draw.textbbox((0, 0), test_line, font=font_small)
                    text_width = bbox[2] - bbox[0]
                    if text_width > max_width and current_line:
                        response_lines.append(current_line)
                        current_line = char
                    else:
                        current_line = test_line
                if current_line:
                    response_lines.append(current_line)
                
                # 绘制回复文字（最多显示2行，避免超出区域）
                max_lines = min(2, len(response_lines))
                for i in range(max_lines):
                    y_pos = robot_y_start + 35 + i * line_height
                    if y_pos < robot_y_end - 10:  # 确保不超出区域
                        draw.text((20, y_pos), response_lines[i], fill=(200, 200, 255), font=font_small)
                
                # 如果回复文字太长，显示省略号
                if len(response_lines) > max_lines:
                    draw.text((20, robot_y_start + 35 + max_lines * line_height), "...", fill=(200, 200, 255), font=font_small)
            
            # 转换回OpenCV格式
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        
        # 显示图像
        cv2.imshow('情感交互系统', frame)
        
        # 检查退出（OpenCV窗口）
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            should_exit = True
    
    # 清理资源
    is_recording = False
    if audio_stream:
        audio_stream.stop_stream()
        audio_stream.close()
    if pyaudio_instance:
        pyaudio_instance.terminate()
    if listener:
        listener.stop()
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    
    # 清理临时文件和缓存文件以保护隐私
    print("\n" + "=" * 60)
    print("正在清理临时文件和缓存文件...")
    print("=" * 60)
    cleanup_temp_files()
    print("=" * 60)
    
    print("\n程序已退出")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被中断")
        # 即使被中断，也清理临时文件
        cleanup_temp_files()
    except Exception as e:
        print(f"\n程序出错：{e}")
        import traceback
        traceback.print_exc()
        # 即使出错，也清理临时文件
        cleanup_temp_files()

