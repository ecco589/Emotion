import os
import time
import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import draw_text, draw_bounding_box, apply_offsets
from utils.preprocessor import preprocess_input

USE_WEBCAM = True # If false, loads video file source

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels_dict = get_labels('fer2013')
emotion_labels = [emotion_labels_dict[i] for i in range(len(emotion_labels_dict))]

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

emotion_window = []
cv2.namedWindow('window_frame')

# 初始化摄像头或视频文件
camera_index = 0
if USE_WEBCAM:
    try:
        with open('../linux_config.txt', 'r') as f:
            for line in f:
                if 'camera_index' in line and '=' in line:
                    camera_index = int(line.split('=')[1].strip())
                    break
    except:
        pass
    
    cap = None
    for idx in [camera_index, 0, 1, 2]:
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, frame = None, None
                for _ in range(10):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"✓ 成功打开摄像头 {idx}")
                        break
                    time.sleep(0.1)
                if ret and frame is not None and frame.size > 0:
                    break
                cap.release()
                cap = None
        if cap is not None and cap.isOpened():
            break
    
    if cap is None or not cap.isOpened():
        print("⚠ 警告：无法打开摄像头，切换到视频文件模式")
        cap = cv2.VideoCapture('./demo/dinner.mp4')
        if not cap.isOpened():
            print("✗ 错误：无法打开视频文件")
            exit(1)
else:
    cap = cv2.VideoCapture('./demo/dinner.mp4')

while cap.isOpened():
    ret, bgr_image = cap.read()
    if not ret:
        break
    
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    current_emotion_probabilities = None

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(np.expand_dims(gray_face, 0), -1)
        emotion_probabilities = emotion_classifier.predict(gray_face, verbose=0)[0]
        current_emotion_probabilities = emotion_probabilities
        emotion_text = emotion_labels[int(np.argmax(emotion_probabilities))]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        emotion_probability = np.max(emotion_probabilities)
        color_map = {'angry': (255, 0, 0), 'sad': (0, 0, 255), 'happy': (255, 255, 0),
                     'surprise': (0, 255, 255), 'neutral': (0, 255, 0), 'disgust': (0, 128, 128),
                     'fear': (128, 0, 128)}
        color = (emotion_probability * np.asarray(color_map.get(emotion_text, (0, 255, 0)))).astype(int).tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)
        break

    # 绘制情绪概率侧边栏
    if len(faces) > 0 and current_emotion_probabilities is not None:
        sidebar_width, start_y, line_height, bar_width, bar_height = 200, 60, 35, 150, 20
        h, w = rgb_image.shape[:2]
        display_image = np.zeros((h, w + sidebar_width, 3), dtype=np.uint8)
        display_image[:, :w] = rgb_image
        
        cv2.rectangle(display_image, (w, 0), (w + sidebar_width, h), (40, 40, 40), -1)
        cv2.rectangle(display_image, (w, 0), (w + sidebar_width, h), (100, 100, 100), 2)
        cv2.putText(display_image, "Emotion Probabilities", (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        emotion_colors = {'angry': (0, 0, 255), 'disgust': (0, 128, 128), 'fear': (128, 0, 128),
                         'happy': (0, 255, 255), 'sad': (255, 0, 0), 'surprise': (255, 255, 0),
                         'neutral': (0, 255, 0)}
        
        for i, label in enumerate(emotion_labels):
            prob = float(current_emotion_probabilities[i])
            color = emotion_colors.get(label, (255, 255, 255))
            y_pos = start_y + i * line_height
            
            cv2.putText(display_image, f"{label}:", (w + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_image, f"{prob:.2f}", (w + sidebar_width - 50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
            bar_x, bar_y = w + 10, y_pos + 5
            bar_fill = int(bar_width * prob)
            cv2.rectangle(display_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
            if bar_fill > 0:
                cv2.rectangle(display_image, (bar_x, bar_y), (bar_x + bar_fill, bar_y + bar_height), color, -1)
            cv2.rectangle(display_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (150, 150, 150), 1)
        
        bgr_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
    else:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
