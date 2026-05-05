import streamlit as st
import cv2
import math
import av
import time
import threading
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import base64
from pathlib import Path
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="長照睡姿固定過久警報系統",
    page_icon="🛌",
    layout="wide"
)

# =========================
# Custom style
# =========================
st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1f3c88;
    margin-bottom: 0.2rem;
}
.sub-text {
    color: #5f6b7a;
    font-size: 1rem;
    margin-bottom: 1.2rem;
}
.metric-card {
    background-color: #f8fbff;
    border: 1px solid #dfe8f3;
    border-radius: 14px;
    padding: 16px 20px;
    text-align: center;
}
.metric-label {
    font-size: 0.95rem;
    color: #6b7280;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1f3c88;
}
.alert-box {
    background-color: #fff1f2;
    border: 1px solid #fda4af;
    color: #b91c1c;
    border-radius: 12px;
    padding: 16px;
    font-size: 1.05rem;
    font-weight: 600;
}
.normal-box {
    background-color: #f0fdf4;
    border: 1px solid #86efac;
    color: #166534;
    border-radius: 12px;
    padding: 16px;
    font-size: 1.05rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Title
# =========================
st.markdown('<div class="main-title">🛌 長照睡姿固定過久警報系統</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">使用影像分析臥床姿勢變化，協助照護員及早發現長時間未翻身狀況。</div>',
    unsafe_allow_html=True
)

# =========================
# Load model
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n-pose.pt")

model = load_model()

# =========================
# Shared state
# =========================
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_posture = "無人躺著"
        self.last_posture = "無人躺著"
        self.start_time = time.time()
        self.duration = 0.0
        self.alarm = False
        self.last_alarm_time = 0.0
        self.alarm_sound_played = False
        self.last_sound_time = 0.0
        self.alarm_acknowledged = False

if "shared_state" not in st.session_state:
    st.session_state.shared_state = AppState()


shared_state = st.session_state.shared_state

if not hasattr(shared_state, "last_sound_time"):
    shared_state.last_sound_time = 0.0

if not hasattr(shared_state, "alarm_acknowledged"):
    shared_state.alarm_acknowledged = False

if "sound_enabled" not in st.session_state:
    st.session_state.sound_enabled = False


# =========================
# Helper functions
# =========================
def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def classify_posture(results):
    current_posture = "無人躺著"

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kps = results[0].keypoints.xy[0]
        conf = results[0].keypoints.conf[0]

        if conf.max() > 0.5 and len(kps) >= 13:
            s_width = dist(kps[5], kps[6])
            t_len = (dist(kps[5], kps[11]) + dist(kps[6], kps[12])) / 2
            is_side = (conf[5] < 0.4 or conf[6] < 0.4) or (t_len > 0 and (s_width / t_len) < 0.5)

            if is_side:
                if (conf[4] + conf[6]) > (conf[3] + conf[5]) + 0.2:
                    current_posture = "左側躺"
                elif (conf[3] + conf[5]) > (conf[4] + conf[6]) + 0.2:
                    current_posture = "右側躺"
                else:
                    current_posture = "右側躺" if dist(kps[0], kps[3]) < dist(kps[0], kps[4]) else "左側躺"
            else:
                current_posture = "仰躺"

    return current_posture
    
def render_loop_alarm(audio_path="alarm.mp3", alarm_on=False):
    """
    警報開啟時，持續播放 loop 音效。
    警報關閉時，不渲染 audio，聲音就會停止。
    """
    if not alarm_on:
        return

    audio_file = Path(__file__).parent / audio_path

    if not audio_file.exists():
        st.warning(f"⚠️ 找不到警鈴音效檔：{audio_path}")
        return

    audio_bytes = audio_file.read_bytes()
    audio_base64 = base64.b64encode(audio_bytes).decode()

    components.html(
        f"""
        <audio id="alarm-audio" autoplay loop controls style="width: 100%;">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>

        <script>
        const audio = document.getElementById("alarm-audio");
        audio.volume = 1.0;

        audio.play().catch(function(error) {{
            console.log("Audio autoplay was blocked:", error);
        }});
        </script>
        """,
        height=60,
    )
# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ 分析設定")

alarm_threshold = st.sidebar.slider(
    "同姿勢維持幾秒觸發警報",
    min_value=3,
    max_value=60,
    value=10,
    step=1
)

show_live_info = st.sidebar.checkbox("顯示即時狀態資訊", value=True)

if st.sidebar.button("🔊 啟用警報聲"):
    st.session_state.sound_enabled = True
    st.sidebar.success("警報聲已啟用")

st.sidebar.markdown("---")
st.sidebar.info("建議先用 webcam 測試：維持相同姿勢超過 10 秒即觸發警報。")
st_autorefresh(interval=1000, key="alarm_refresh")
# =========================
# Video processor
# =========================
class PoseVideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, verbose=False)
        current_posture = classify_posture(results)

        now = time.time()

        with shared_state.lock:
            if current_posture == shared_state.last_posture:
                shared_state.duration = now - shared_state.start_time
            else:
                shared_state.last_posture = current_posture
                shared_state.current_posture = current_posture
                shared_state.start_time = now
                shared_state.duration = 0.0
                shared_state.alarm = False
                shared_state.alarm_sound_played = False
                shared_state.alarm_acknowledged = False

            if (
                shared_state.duration >= alarm_threshold
                and current_posture != "無人躺著"
                and not shared_state.alarm_acknowledged
            ):
                shared_state.alarm = True
                shared_state.last_alarm_time = now
            else:
                if current_posture == "無人躺著" or shared_state.alarm_acknowledged:
                    shared_state.alarm = False

            shared_state.current_posture = current_posture

        annotated = results[0].plot()

        with shared_state.lock:
            info_text = f"姿勢: {shared_state.current_posture} | 持續時間: {int(shared_state.duration)} 秒"

            cv2.rectangle(annotated, (20, 20), (760, 70), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                info_text,
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            if shared_state.alarm:
                cv2.rectangle(
                    annotated,
                    (0, 0),
                    (annotated.shape[1], annotated.shape[0]),
                    (0, 0, 255),
                    10
                )
                cv2.putText(
                    annotated,
                    "ALARM",
                    (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 0, 255),
                    4,
                    cv2.LINE_AA
                )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# =========================
# Layout
# =========================
left_col, right_col = st.columns([1.15, 1.4])

with left_col:
    st.subheader("1. 即時影像監測")

    webrtc_streamer(
        key="pose-monitor",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=PoseVideoProcessor,
        async_processing=True,
    )

with right_col:
    st.subheader("2. 摘要資訊")

    with shared_state.lock:
        posture_now = shared_state.current_posture
        duration_now = int(shared_state.duration)
        alarm_now = shared_state.alarm

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">目前姿勢</div>
                <div class="metric-value">{posture_now}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">持續時間</div>
                <div class="metric-value">{duration_now} 秒</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        alarm_text = "是" if alarm_now else "否"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">是否警報</div>
                <div class="metric-value">{alarm_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("3. 警報摘要")

    if alarm_now:
        st.markdown(
            f"""
            <div class="alert-box">
                🚨 偵測到姿勢持續超過 {alarm_threshold} 秒，請協助翻身。
            </div>
            """,
            unsafe_allow_html=True
        )
    
        if st.button("✅ 確認此資訊", type="primary"):
            with shared_state.lock:
                shared_state.alarm_acknowledged = True
                shared_state.alarm = False
                shared_state.last_sound_time = time.time()
            st.rerun()
    
        if st.session_state.sound_enabled:
            render_loop_alarm("alarm.mp3", alarm_on=True)
        else:
            st.warning("🔇 請先按左側「啟用警報聲」，瀏覽器才比較可能允許播放聲音。")
