"""
真实交互Demo - 按键录音 + 情绪检测 + LLM生成
按空格键开始录音和情绪检测，松开停止，自动生成回复
"""
import os
import sys
# Suppress all warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'  # Suppress OpenCV warnings
os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress Python warnings
# Suppress ALSA warnings
os.environ['ALSA_CARD'] = '0'
import warnings
warnings.filterwarnings('ignore')
import cv2
try:
    cv2.setLogLevel(0)  # 0 = SILENT, suppress OpenCV warnings
except:
    pass  # If setLogLevel doesn't work, continue anyway
import numpy as np
import time
import wave
import glob
import re
import uuid
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
# 兼容新旧版本Keras：优先使用tensorflow.keras，否则使用旧版keras
try:
    from tensorflow.keras.models import load_model
    print("✓ 使用 tensorflow.keras")
except ImportError:
    try:
        from keras.models import load_model
        print("✓ 使用 keras (旧版本)")
    except ImportError:
        print("✗ 错误：未找到 keras 或 tensorflow.keras")
        raise
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
recording_start_time = None  # 录音开始的时间戳
current_emotion_label = "neutral"  # 当前显示的情绪标签
current_emotion_conf = 0.5  # 当前显示的情绪置信度

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
space_pressed_stable = False  # 稳定的按键状态，用于避免事件处理延迟导致的状态抖动
listener = None

# 线程控制
audio_thread_running = False  # 控制录音线程是否运行
audio_thread = None  # 录音线程对象

# 界面显示状态
robot_response_display = None  # 存储最新的机器人回复用于显示
user_voice_display = None  # 存储用户语音转录文字用于显示
display_timeout = 0  # 显示超时时间
user_display_timeout = 0  # 用户语音显示超时时间

# ==================== 初始化函数 ====================

def load_linux_config():
    """读取Linux配置文件"""
    config_file = "linux_config.txt"
    config = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
        except:
            pass
    return config

def init_emotion_module():
    """初始化情感检测模块"""
    global emotion_classifier, face_cascade, emotion_target_size, cap
    
    print("初始化情感检测模块...")
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model(emotion_model_path, compile=False)
    emotion_target_size = emotion_classifier.input_shape[1:3]
    
    # 根据操作系统选择摄像头
    system = platform.system()
    if system == "Linux":
        # Linux: 从配置文件读取摄像头索引
        config = load_linux_config()
        camera_index = config.get('camera_index', '0')
        try:
            camera_index = int(camera_index)
        except:
            camera_index = 0
        
        # 尝试打开指定的摄像头，使用V4L2后端
        cap = None
        backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
        
        for backend in backends_to_try:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                # 等待更多帧让摄像头初始化（虚拟机需要更长时间）
                ret, test_frame = None, None
                for _ in range(30):  # 增加等待时间
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        break
                    time.sleep(0.1)  # 等待100ms
                if ret and test_frame is not None and test_frame.size > 0:
                    print(f"✓ 成功打开摄像头 {camera_index} (后端: {backend})")
                    break
                else:
                    cap.release()
                    cap = None
        
        if not cap or not cap.isOpened():
            # 如果指定摄像头失败，尝试其他索引
            for i in range(3):
                if i != camera_index:
                    for backend in backends_to_try:
                        cap = cv2.VideoCapture(i, backend)
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            ret, test_frame = None, None
                            for _ in range(30):
                                ret, test_frame = cap.read()
                                if ret and test_frame is not None and test_frame.size > 0:
                                    break
                                time.sleep(0.1)
                            if ret and test_frame is not None and test_frame.size > 0:
                                print(f"⚠ 警告：配置的摄像头{camera_index}无法打开，使用摄像头{i}")
                                break
                            else:
                                cap.release()
                                cap = None
                    if cap and cap.isOpened():
                        break
        
        if not cap or not cap.isOpened():
            print("⚠ 警告：无法打开摄像头或读取视频流")
            print("程序将在测试模式下运行（使用测试图像）")
            print("提示：")
            print("  1. 摄像头是否已连接到虚拟机")
            print("  2. 在VMware中：虚拟机设置 -> USB控制器 -> 启用USB 3.0/3.1")
            print("  3. 将摄像头连接到虚拟机（VMware菜单：虚拟机 -> 可移动设备 -> 摄像头）")
            # 设置为None，但继续运行（返回True）
            cap = None
            print("✓ 情感检测模块初始化完成（测试模式）")
            return True  # 继续运行，即使没有摄像头
    elif system == "Darwin":  # macOS
        cap = cv2.VideoCapture(1)  # MacBook前置摄像头
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)  # 默认摄像头
        if not cap.isOpened():
            print("✗ 错误：无法打开摄像头")
            return False
    else:  # Windows 或其他
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("✗ 错误：无法打开摄像头")
            return False
    
    print("✓ 情感检测模块初始化完成")
    return True

def get_baidu_access_token(api_key=None, secret_key=None):
    """获取百度ASR Access Token"""
    url = "https://aip.baidubce.com/oauth/2.0/token"
    
    # 如果未提供参数，使用全局变量
    if api_key is None:
        api_key = BAIDU_ASR_API_KEY
    if secret_key is None:
        secret_key = BAIDU_ASR_SECRET_KEY
    
    # 如果全局变量为空，尝试从环境变量读取
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
        return False
    try:
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        pyaudio_instance = pyaudio.PyAudio()
        sys.stderr.close()
        sys.stderr = old_stderr
        return True
    except:
        if 'old_stderr' in locals():
            sys.stderr = old_stderr
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
        # 调试输出：未检测到人脸（降低频率）
        if len(emotion_window) % 30 == 0:  # 每30次输出一次
            print(f"  [情感检测] 未检测到人脸，返回neutral 0.5")
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
        # 调试输出：低置信度警告
        if len(emotion_window) % 10 == 0:  # 每10次输出一次，避免刷屏
            print(f"  [情感检测] 置信度过低({emotion_probability:.2f})，强制设为neutral 0.5")
    
    return (emotion_mode, emotion_probability)

def record_audio():
    """录音线程函数"""
    global is_recording, recording_frames, audio_stream, pyaudio_instance, audio_thread_running
    
    if not PYAUDIO_AVAILABLE:
        return
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    # 防抖机制：只有当 is_recording 持续为 False 多次循环后才真正停止
    stop_counter = 0
    STOP_THRESHOLD = 5  # 连续5次检测到 False 才停止（约50ms）
    
    # 限制recording_frames的最大大小，防止内存泄漏（约10秒的录音数据）
    MAX_FRAMES = 1000  # 约10秒 @ 16kHz
    
    while audio_thread_running:
        if is_recording and audio_stream is None:
            # 开始录音（按下空格键时）
            stop_counter = 0  # 重置计数器
            try:
                audio_stream = pyaudio_instance.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK
                )
                recording_frames = []  # 清空之前的录音数据
            except:
                audio_stream = None
        
        elif is_recording and audio_stream:
            # 录音中，持续收集数据（整个按下期间）
            stop_counter = 0  # 重置计数器
            try:
                data = audio_stream.read(CHUNK, exception_on_overflow=False)
                recording_frames.append(data)
                # 限制recording_frames大小，防止内存泄漏
                if len(recording_frames) > MAX_FRAMES:
                    recording_frames.pop(0)  # 移除最旧的数据
            except Exception as e:
                # 如果读取失败，不要关闭流，而是继续尝试
                # 这样可以避免频繁的"录音结束"和"录音开始"消息
                # 只记录错误，不中断录音流程
                pass  # 静默处理，继续下一次循环尝试读取
        
        elif not is_recording and audio_stream:
            # 检测到停止信号，使用防抖机制
            stop_counter += 1
            if stop_counter >= STOP_THRESHOLD:
                # 只有当连续多次检测到停止信号时才真正停止录音
                try:
                    audio_stream.stop_stream()
                    audio_stream.close()
                    audio_stream = None
                    stop_counter = 0
                except:
                    audio_stream = None
                    stop_counter = 0
        
        time.sleep(0.01)  # 避免CPU占用过高
    
    # 线程退出前，确保关闭音频流
    if audio_stream:
        try:
            try:
                # 某些PyAudio版本可能没有is_active方法
                if hasattr(audio_stream, 'is_active') and audio_stream.is_active():
                    audio_stream.stop_stream()
            except:
                # 即使is_active检查失败，也尝试停止流
                try:
                    audio_stream.stop_stream()
                except:
                    pass
            audio_stream.close()
            audio_stream = None
        except:
            audio_stream = None

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
        
        if len(audio_data) < 1000:
            return "未识别到语音"
        
        # 使用百度SDK（如果可用且有API_KEY和SECRET_KEY）
        try:
            if BAIDU_ASR_API_KEY and BAIDU_ASR_SECRET_KEY:
                try:
                    from aip import AipSpeech
                    aip_client = AipSpeech("", BAIDU_ASR_API_KEY, BAIDU_ASR_SECRET_KEY)
                    result = aip_client.asr(audio_data, 'wav', 16000, {'dev_pid': 1537})
                    if result and 'result' in result and len(result['result']) > 0:
                        return result['result'][0]
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
            
            # 如果token是bce-v3格式，需要先获取access_token
            token = asr_client.get("token", "")
            if token.startswith("bce-v3/"):
                print("检测到BCE签名token，尝试使用API_KEY获取access_token...")
                # 优先使用代码中配置的API_KEY和SECRET_KEY
                api_key = BAIDU_ASR_API_KEY
                secret_key = BAIDU_ASR_SECRET_KEY
                
                # 如果代码中未配置，尝试从环境变量获取
                if not api_key or not secret_key:
                    api_key = os.getenv("BAIDU_ASR_API_KEY", "")
                    secret_key = os.getenv("BAIDU_ASR_SECRET_KEY", "")
                
                if api_key and secret_key:
                    access_token = get_baidu_access_token(api_key, secret_key)
                    if access_token:
                        token = access_token
                    else:
                        return "未识别到语音"
                else:
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
                
                response = requests.post(url, json=payload, headers=headers, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    if result.get('err_no') == 0 and result.get('result'):
                        if len(result['result']) > 0:
                            return result['result'][0]
            except:
                try:
                    files = {'audio': ('audio.wav', audio_data, 'audio/wav')}
                    data = {'format': 'wav', 'rate': '16000', 'channel': '1', 'cuid': 'VPMAnb5S3dfr6RPD67qvzmNBO850fDTc', 'dev_pid': '1537', 'token': token}
                    response = requests.post(url, files=files, data=data, timeout=10)
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('err_no') == 0 and result.get('result'):
                            if len(result['result']) > 0:
                                return result['result'][0]
                except:
                    pass
        
        return "未识别到语音"
    except:
        return "未识别到语音"

def init_robot_personality():
    """初始化机器人个性配置"""
    global robot_personality
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
        response = requests.post(url, json=data, headers=headers, timeout=15)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]
            llm_text = ""
            
            if "content" in message and message["content"] and message["content"].strip():
                llm_text = message["content"].strip()
            elif "reasoning_content" in message and message["reasoning_content"]:
                reasoning_text = message["reasoning_content"].strip()
                
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
                                    output_lines = lines[i:i+3]
                                    found_label = True
                                    break
                                except:
                                    pass
                    if found_label:
                        break
                
                # 方法2：如果没找到，尝试从推理内容中提取LLM想出的回复
                if not output_lines:
                    # 查找推理过程中提到的回复文字（通常在引号内或"比如"后面）
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
                        
                        # 查找情绪程度（更精确的匹配）
                        level_matches = re.findall(r'程度\s*([0-9.]+)', reasoning_text)
                        if not level_matches:
                            # 尝试查找"0.7"、"0.8"等数字
                            level_matches = re.findall(r'\b([0-9]\.[0-9])\b', reasoning_text)
                        
                        if level_matches:
                            try:
                                level_from_reasoning = float(level_matches[-1])
                            except:
                                pass
                        else:
                            # 如果没有找到，使用默认值
                            level_from_reasoning = 0.7
                        
                        if potential_response and len(potential_response) > 5:
                            output_lines = [emotion_from_reasoning, str(level_from_reasoning), potential_response]
                
                if output_lines and len(output_lines) >= 3:
                    llm_text = "\n".join(output_lines)
                else:
                    llm_text = reasoning_text
            
            if not llm_text:
                raise Exception("LLM输出为空")
        else:
            raise Exception("API返回格式错误")
        
        # 解析为3类结果
        # 如果LLM输出了推理过程，需要提取真正的3行输出
        lines_raw = llm_text.split("\n")
        lines = [line.strip() for line in lines_raw if line.strip()]
        
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
                quotes = re.findall(r'[""](.*?)[""]', response_text)
                if quotes:
                    response_text = quotes[0]
                else:
                    # 如果没有引号，提取前50个字符
                    response_text = response_text[:50]
        
        valid_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        if emotion_label not in valid_labels:
            emotion_label = "neutral"
        
        return {
            "emotion_label": emotion_label,
            "emotion_level": emotion_level,
            "response_text": response_text
        }
    except:
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
            except Exception:
                pass
        
        # 清理Python缓存文件
        import shutil
        cache_dirs = glob.glob("__pycache__") + glob.glob("**/__pycache__", recursive=True)
        for cache_dir in cache_dirs:
            try:
                if os.path.isdir(cache_dir):
                    shutil.rmtree(cache_dir)
            except Exception:
                pass
        
        # 清理其他可能的临时文件
        temp_files = glob.glob("*.tmp") + glob.glob("*.cache")
        for file in temp_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except Exception:
                pass
    except Exception:
        pass

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
                return None
        
        # 百度TTS API地址
        url = "https://tsn.baidu.com/text2audio"
        
        # 参数准备
        # tex需要2次urlencode（根据百度文档要求）
        # 注意：不能使用requests的data参数自动编码，因为需要tex单独2次编码
        tex_encoded = urllib.parse.quote(text, safe='')
        tex_encoded = urllib.parse.quote(tex_encoded, safe='')
        
        # 用户唯一标识（使用UUID）
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
            return audio_filename
        else:
            # 合成失败，返回错误信息
            try:
                error_info = response.json()
                err_no = error_info.get('err_no', '未知')
                err_msg = error_info.get('err_msg', '未知错误')
            except:
                pass
        return None
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def play_audio(audio_file):
    """播放音频文件
    
    Args:
        audio_file: 音频文件路径
    """
    if not audio_file or not os.path.exists(audio_file):
        return False
    
    try:
        system = platform.system()
        
        if system == "Darwin":  # macOS
            # 使用afplay命令播放
            subprocess.Popen(['afplay', audio_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        elif system == "Linux":
            # 尝试使用mpg123（支持mp3）或ffplay（支持多种格式）
            try:
                # 检查文件扩展名
                if audio_file.endswith('.mp3'):
                    # MP3文件使用mpg123，等待播放完成
                    proc = subprocess.Popen(['mpg123', '-q', audio_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    proc.wait()  # 等待播放完成
                else:
                    # 其他格式尝试ffplay，等待播放完成
                    proc = subprocess.Popen(['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', audio_file], 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    proc.wait()  # 等待播放完成
                return True
            except FileNotFoundError:
                try:
                    # 如果mpg123/ffplay不可用，尝试aplay（仅支持wav）
                    if audio_file.endswith('.wav'):
                        proc = subprocess.Popen(['aplay', audio_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        proc.wait()  # 等待播放完成
                        return True
                    else:
                        return False
                except:
                    return False
        elif system == "Windows":
            # 使用Windows的start命令
            subprocess.Popen(['start', audio_file], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        else:
            return False
    except:
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
    global space_pressed, space_pressed_stable
    try:
        if key == keyboard.Key.space:
            space_pressed = True
            space_pressed_stable = True  # 立即更新稳定状态
    except:
        pass

def on_key_release(key):
    """按键释放回调"""
    global space_pressed, space_pressed_stable
    try:
        if key == keyboard.Key.space:
            space_pressed = False
            space_pressed_stable = False  # 立即更新稳定状态
        elif key == keyboard.Key.esc:
            return False  # 停止监听
    except:
        pass

def main():
    """主程序"""
    global is_recording, recording_frames, cap, user_voice_display, user_display_timeout, space_pressed, space_pressed_stable, listener, key_release_counter, recording_start_time, current_emotion_label, current_emotion_conf, robot_response_display, display_timeout, audio_thread_running, audio_thread, audio_stream, pyaudio_instance
    
    # 首先清理可能残留的资源（防止之前运行留下的资源）
    try:
        is_recording = False
        audio_thread_running = False
        if audio_stream:
            try:
                try:
                    if hasattr(audio_stream, 'is_active') and audio_stream.is_active():
                        audio_stream.stop_stream()
                except:
                    pass
                audio_stream.close()
            except:
                pass
            audio_stream = None
        if pyaudio_instance:
            try:
                pyaudio_instance.terminate()
            except:
                pass
            pyaudio_instance = None
        if listener:
            try:
                listener.stop()
            except:
                pass
            listener = None
        if cap:
            try:
                cap.release()
            except:
                pass
            cap = None
        recording_frames = []
    except:
        pass
    
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
        print("⚠ 警告：摄像头初始化失败，将在测试模式下运行")
        # 不返回，继续运行
    
    asr_client = init_asr()
    if not init_audio():
        print("错误：音频录制初始化失败")
        return
    
    # 启动录音线程
    audio_thread_running = True
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
    # 设置窗口大小（1280x720）
    cv2.resizeWindow('情感交互系统', 1280, 720)
    
    last_key_state = False
    emotions_during_recording = []
    should_exit = False
    key_release_counter = 0  # 按键松开计数器，用于防抖
    
    while not should_exit:
        # 读取摄像头帧或使用测试图像
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                # 如果无法读取帧，创建测试图像
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera not available", (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, "Using test mode", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press Q to exit", (50, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        else:
            # 摄像头未初始化，使用测试图像
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not available", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Using test mode", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press Q to exit", (50, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 检测按键状态（使用稳定的按键状态，避免事件处理延迟导致的状态抖动）
        if KEYBOARD_AVAILABLE:
            # 直接使用稳定状态，不要同步（on_key_press/release已经更新了）
            # 移除同步逻辑，避免因为 space_pressed 的短暂变化导致误判
            current_key_state = space_pressed_stable
        else:
            # 使用OpenCV的waitKey作为备用方案
            key = cv2.waitKey(1) & 0xFF
            current_key_state = (key == ord(' ') or key == 32)
            if key == ord('q') or key == ord('Q'):
                should_exit = True
        
        # 检查空格键状态（对讲机模式：按下录音，松开停止）
        # 使用防抖机制：只有当按键状态持续为False一段时间后才认为真正松开
        if current_key_state:
            # 空格键被按下
            if not last_key_state:
                # 刚按下空格键，立即开始录音和情绪检测
                is_recording = True
                recording_start_time = time.time()  # 记录开始录音的时间戳
                emotions_during_recording = []
                # 初始化情绪显示值
                current_emotion_label = "neutral"
                current_emotion_conf = 0.5
                # 注意：不要在这里清空 recording_frames，让 record_audio 线程自己管理
                # recording_frames = []  # 移除这行，避免与录音线程冲突
                print("\n" + "=" * 60)
                print("【开始录音】按住空格键说话，松开停止...")
                print("=" * 60)
            
            last_key_state = True
            key_release_counter = 0  # 重置松开计数器（全局变量）
            
            # 按下期间持续录音（is_recording已经在record_audio线程中处理）
            # 实时检测情绪并更新显示（每帧都检测并更新显示，确保实时性）
            # 每帧都检测情绪，但每2帧记录一次到列表（减少列表大小）
            emotion_label, emotion_conf = detect_emotion_from_frame(frame)
            # 每帧都更新显示值，确保实时显示
            current_emotion_label = emotion_label
            current_emotion_conf = emotion_conf
            
            # 每2帧记录一次到列表（用于最终计算）
            # 使用帧计数器而不是列表长度来判断，确保每2帧记录一次
            frame_count = len(emotions_during_recording)
            if frame_count % 2 == 0:
                emotions_during_recording.append((emotion_label, emotion_conf))
            
            # 调试输出：每30帧输出一次，避免刷屏
            if frame_count > 0 and frame_count % 30 == 0:
                print(f"  [实时检测] 情绪: {emotion_label}, 置信度: {emotion_conf:.2f}, 已记录: {len(emotions_during_recording)}次")
        
        elif last_key_state:
            # 检测到按键状态变为False，但使用防抖机制确认
            # 只有当连续多次检测到False时才认为真正松开
            key_release_counter += 1
            
            # 只有当连续10次（约100ms，假设每帧10ms）检测到松开时才真正停止录音
            # 这样可以避免因为按键检测的短暂抖动导致误判
            if key_release_counter >= 10:
                # 真正松开空格键，停止录音并处理
                is_recording = False
                last_key_state = False
                key_release_counter = 0
                
                time.sleep(0.1)
                
                # 计算最终情绪（使用mode）
                print(f"\n【情感检测调试信息】")
                print(f"  列表长度：{len(emotions_during_recording)}")
                print(f"  列表内容：{emotions_during_recording}")
                
                if emotions_during_recording:
                    emotion_list = [emo for emo, conf in emotions_during_recording]
                    try:
                        final_emotion = mode(emotion_list)
                    except:
                        final_emotion = emotions_during_recording[-1][0]
                    
                    # 计算该情绪的平均置信度
                    confs = [conf for emo, conf in emotions_during_recording if emo == final_emotion]
                    final_conf = sum(confs) / len(confs) if confs else 0.5
                    
                    # 调试输出：显示检测到的所有情绪
                    print(f"  检测次数：{len(emotions_during_recording)}")
                    print(f"  所有检测结果：{emotions_during_recording}")
                    print(f"  最终情绪：{final_emotion}")
                    print(f"  最终置信度：{final_conf:.2f}")
                else:
                    # 如果列表为空，使用当前显示的情绪值（这是实时更新的）
                    final_emotion = current_emotion_label
                    final_conf = current_emotion_conf
                    print(f"\n【警告】录音期间列表为空，使用当前显示值：{final_emotion} {final_conf:.2f}")
                    print(f"  当前显示的情绪：{current_emotion_label}")
                    print(f"  当前显示的置信度：{current_emotion_conf:.2f}")
                
                # 保存录音文件
                if recording_frames:
                    audio_file = save_audio_to_file(recording_frames)
                    
                    # 检查文件是否保存成功
                    if os.path.exists(audio_file):
                        file_size = os.path.getsize(audio_file)
                        voice_text = recognize_speech(audio_file, asr_client)
                        user_voice_display = voice_text
                        user_display_timeout = time.time() + 30
                    else:
                        voice_text = "未识别到语音"
                        user_voice_display = voice_text
                        user_display_timeout = time.time() + 30
                    
                    if voice_text == "未识别到语音":
                        robot_output = {
                            "emotion_label": "neutral",
                            "emotion_level": 0.7,
                            "response_text": "抱歉，我没有听清楚，请再说一遍。"
                        }
                        robot_response_display = robot_output
                        display_timeout = time.time() + 10
                    else:
                        user_sync_data = {
                            "voice_text": voice_text,
                            "emotion_label": final_emotion,
                            "emotion_conf": final_conf
                        }
                        print(f"\n【传递给LLM的数据】")
                        print(f"  语音：{voice_text}")
                        print(f"  情绪：{final_emotion}")
                        print(f"  置信度：{final_conf:.2f}")
                        robot_output = call_llm(user_sync_data, short_term_memory)
                    
                    if voice_text != "未识别到语音":
                        update_memory(user_sync_data, robot_output)
                    
                    tts_audio_file = text_to_speech(robot_output['response_text'])
                    if tts_audio_file:
                        play_audio(tts_audio_file)
                        try:
                            if os.path.exists(tts_audio_file):
                                os.remove(tts_audio_file)
                        except:
                            pass
                    
                    # 清空录音帧
                    recording_frames = []
            else:
                pass
            
            emotions_during_recording = []
        
        # 在界面上显示信息（使用PIL绘制中文和颜文字）
        # 用户语音和AI回复同时显示在一个界面中
        show_user_voice = user_voice_display and time.time() < user_display_timeout
        show_robot_response = robot_response_display and time.time() < display_timeout
        
        if show_user_voice or show_robot_response:
            
            # 转换为PIL Image以支持中文和颜文字
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            
            # 尝试加载中文字体（根据操作系统选择）
            system = platform.system()
            font_large = None
            font_small = None
            font_medium = None
            
            if system == "Darwin":  # macOS
                font_paths = [
                    "/System/Library/Fonts/PingFang.ttc",
                    "/System/Library/Fonts/STHeiti Medium.ttc"
                ]
            elif system == "Linux":  # Linux
                font_paths = [
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                    "/usr/share/fonts/truetype/arphic/uming.ttc",
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                ]
                # 确保字体文件存在
                font_paths = [f for f in font_paths if os.path.exists(f)]
            else:  # Windows 或其他
                font_paths = [
                    "C:/Windows/Fonts/simhei.ttf",
                    "C:/Windows/Fonts/msyh.ttc"
                ]
            
            # 尝试加载字体
            for font_path in font_paths:
                try:
                    if os.path.exists(font_path):
                        font_large = ImageFont.truetype(font_path, 24)
                        font_small = ImageFont.truetype(font_path, 20)
                        font_medium = ImageFont.truetype(font_path, 22)
                        break
                except:
                    continue
            
            # 如果所有字体都加载失败，使用默认字体
            if font_large is None:
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
            
            # 重新加载字体（支持中文）
            font_medium = None
            font_small = None
            system = platform.system()
            
            if system == "Darwin":  # macOS
                font_paths = [
                    "/System/Library/Fonts/PingFang.ttc",
                    "/System/Library/Fonts/STHeiti Medium.ttc"
                ]
            elif system == "Linux":  # Linux
                font_paths = [
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                    "/usr/share/fonts/truetype/arphic/uming.ttc",
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
                ]
                font_paths = [f for f in font_paths if os.path.exists(f)]
            else:  # Windows
                font_paths = [
                    "C:/Windows/Fonts/simhei.ttf",
                    "C:/Windows/Fonts/msyh.ttc"
                ]
            
            for font_path in font_paths:
                try:
                    if os.path.exists(font_path):
                        font_medium = ImageFont.truetype(font_path, 22)
                        font_small = ImageFont.truetype(font_path, 20)
                        break
                except:
                    continue
            
            if font_medium is None:
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
        
        # 如果正在录音，绘制录音状态（简单直接，只要is_recording为True就显示）
        if is_recording and recording_start_time is not None:
            # 计算录音时长
            recording_time = time.time() - recording_start_time
            
            # 使用稳定的情绪显示值（在检测时已更新，不会回到默认值）
            emotion_label = current_emotion_label
            emotion_conf = current_emotion_conf
            
            # 绘制录音状态（始终显示在最上层）
            # 使用最新的情绪值实时显示
            cv2.putText(frame, f"Emotion: {emotion_label} ({emotion_conf:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"RECORDING... ({recording_time:.1f}s) [Release SPACE to stop]", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # 显示录音指示器
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        else:
            # 不在录音时重置情绪显示值
            current_emotion_label = "neutral"
            current_emotion_conf = 0.5
        
        # 显示图像
        cv2.imshow('情感交互系统', frame)
        
        # 检查退出（OpenCV窗口）
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            should_exit = True
    
    # 清理资源（按顺序，确保正确关闭）
    print("\n正在清理资源...")
    
    # 1. 停止录音线程
    is_recording = False
    audio_thread_running = False
    
    # 2. 等待录音线程结束（最多等待2秒）
    if audio_thread and audio_thread.is_alive():
        audio_thread.join(timeout=2.0)
    
    # 3. 关闭音频流
    if audio_stream:
        try:
            try:
                # 某些PyAudio版本可能没有is_active方法
                if hasattr(audio_stream, 'is_active') and audio_stream.is_active():
                    audio_stream.stop_stream()
            except:
                pass
            audio_stream.close()
            audio_stream = None
        except Exception as e:
            audio_stream = None
    
    # 4. 终止PyAudio实例
    if pyaudio_instance:
        try:
            pyaudio_instance.terminate()
            pyaudio_instance = None
        except Exception as e:
            pass
    
    # 5. 停止键盘监听
    if listener:
        try:
            listener.stop()
            listener = None
        except Exception as e:
            pass
    
    # 6. 释放摄像头
    if cap:
        try:
            cap.release()
            cap = None
        except Exception as e:
            pass
    
    # 7. 关闭OpenCV窗口
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    # 8. 清空录音数据
    recording_frames = []
    
    # 9. 清理临时文件和缓存文件以保护隐私
    cleanup_temp_files()
    
    print("资源清理完成")
    print("\n程序已退出")

def cleanup_all_resources():
    """清理所有资源的辅助函数"""
    global is_recording, audio_thread_running, audio_stream, pyaudio_instance, listener, cap, recording_frames
    
    try:
        is_recording = False
        audio_thread_running = False
        
        if audio_stream:
            try:
                try:
                    # 某些PyAudio版本可能没有is_active方法
                    if hasattr(audio_stream, 'is_active') and audio_stream.is_active():
                        audio_stream.stop_stream()
                except:
                    pass
                audio_stream.close()
                audio_stream = None
            except:
                audio_stream = None
        
        if pyaudio_instance:
            try:
                pyaudio_instance.terminate()
                pyaudio_instance = None
            except:
                pass
        
        if listener:
            try:
                listener.stop()
                listener = None
            except:
                pass
        
        if cap:
            try:
                cap.release()
                cap = None
            except:
                pass
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        recording_frames = []
        cleanup_temp_files()
    except:
        pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被中断")
        cleanup_all_resources()
    except Exception as e:
        print(f"\n程序出错：{e}")
        import traceback
        traceback.print_exc()
        cleanup_all_resources()

