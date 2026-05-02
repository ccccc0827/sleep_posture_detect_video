!pip install streamlit-webrtc av ultralytics opencv-python-headless -q
%%writefile app.py
import streamlit as st
import cv2
import math
import av
import time
import sys
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from IPython.display import Audio, display

def log_now(msg):
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()

st.set_page_config(page_title="10秒姿勢監測系統", layout="centered")
st.title("🛌 睡姿定時監測系統 (10秒判定)")

@st.cache_resource
def load_model():
    return YOLO("yolov8n-pose.pt")
model = load_model()

class PoseVideoProcessor:
    def __init__(self):
        self.last_posture = "無人躺著"
        self.start_time = time.time()  # 記錄目前姿勢開始的時間
        self.last_print_time = time.time()
        log_now("🟢 系統啟動：開始監測姿勢...")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, verbose=False)
        current_posture = "無人躺著"

        # 1. AI 姿勢辨識邏輯
        if results[0].keypoints is not None and len(results[0].keypoints.data[0]) > 0:
            kps = results[0].keypoints.xy[0]
            conf = results[0].keypoints.conf[0]
            if conf.max() > 0.5 and len(kps) >= 13:
                def dist(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                s_width = dist(kps[5], kps[6])
                t_len = (dist(kps[5], kps[11]) + dist(kps[6], kps[12])) / 2
                is_side = (conf[5] < 0.4 or conf[6] < 0.4) or (t_len > 0 and (s_width / t_len) < 0.5)

                if is_side:
                    if (conf[4] + conf[6]) > (conf[3] + conf[5]) + 0.2: current_posture = "有人向左側躺"
                    elif (conf[3] + conf[5]) > (conf[4] + conf[6]) + 0.2: current_posture = "有人向右側躺"
                    else: current_posture = "有人向右側躺" if dist(kps[0], kps[3]) < dist(kps[0], kps[4]) else "有人向左側躺"
                else:
                    current_posture = "有人仰躺中"

        # 2. 時間判定邏輯 (核心功能)
        now = time.time()

        if current_posture == self.last_posture:
            # 姿勢沒變，計算持續多久了
            duration = now - self.start_time
            if duration >= 10.0:
                # 只有滿 10 秒才印出「確認」訊息
                if now - self.last_print_time >= 3.0: # 避免印太快
                    log_now(f"⚠️ [警告] 持續【{current_posture}】已超過 10 秒！")
                    display(Audio('/content/ElevenLabs_警报器.mp3'))
                    self.last_print_time = now
        else:
            # 姿勢變了，重新計時
            log_now(f"🔄 狀態改變：{self.last_posture} ➡️ {current_posture}")
            self.last_posture = current_posture
            self.start_time = now

        return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")

webrtc_streamer(
    key="pose-10s",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_processor_factory=PoseVideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
import urllib.request
import time
!pkill -f streamlit
!pkill -f localtunnel

# 1. 取得密碼
external_ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n")
print(f"🔑 密碼: {external_ip}")

# 2. 啟動隧道
get_ipython().system_raw('npx localtunnel --port 8501 > url.txt 2>&1 &')
time.sleep(5)
with open('url.txt', 'r') as f:
    print(f"👉 網址: {f.read()}")

# 3. 直接啟動
!streamlit run app.py --server.enableCORS false
