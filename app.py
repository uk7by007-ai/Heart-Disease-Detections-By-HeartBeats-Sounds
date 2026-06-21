import streamlit as st
import numpy as np
import librosa
import librosa.display
import io
import os
import uuid
import tempfile
from datetime import datetime
import pandas as pd
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use("Agg")

# WebRTC for Live Recording
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# FPDF for Report Generation
try:
    from fpdf import FPDF
except ImportError:
    st.error("Please run `pip install fpdf` in your terminal to enable PDF Generation.")
    FPDF = None

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from prediction import load_model, predict_heartbeat

# =====================================================================
# PAGE CONFIG & ROBUST FOLDER MANAGEMENT
# =====================================================================
st.set_page_config(
    page_title="CardioGuard AI - Heart Disease Detection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

os.makedirs("recordings", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# =====================================================================
# CUSTOM CSS
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #060910;
    color: #e2e8f0;
}

.header-wrap { text-align: center; padding: 3rem 0 1.5rem; }
.header-wrap .badge {
    display: inline-block; background: rgba(229,62,62,0.12); border: 1px solid rgba(229,62,62,0.35);
    color: #fc8181; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; padding: 0.3rem 0.9rem; border-radius: 999px; margin-bottom: 1rem;
}
.header-wrap h1 { font-size: 3rem; font-weight: 800; color: #fff; margin: 0; line-height: 1.15; }
.header-wrap p { color: #718096; font-size: 1rem; margin-top: 0.6rem; }

.divider { border: none; border-top: 1px solid #1a202c; margin: 1.8rem 0; }
.section-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #4a5568; margin-bottom: 0.6rem; }

.dx-banner { border-radius: 16px; padding: 2rem 2.2rem; display: flex; align-items: center; gap: 1.6rem; margin: 1.2rem 0; }
.dx-banner.normal   { background: linear-gradient(135deg,#0d2b1e 0%,#0a1f16 100%); border: 1px solid #276749; }
.dx-banner.murmur   { background: linear-gradient(135deg,#2b0d0d 0%,#1a0808 100%); border: 1px solid #9b2c2c; }
.dx-banner.extra    { background: linear-gradient(135deg,#2b1e0d 0%,#1a1208 100%); border: 1px solid #975a16; }
.dx-banner.artifact { background: linear-gradient(135deg,#0d1e2b 0%,#08121a 100%); border: 1px solid #2b6cb0; }
.dx-icon { font-size: 3.2rem; line-height: 1; flex-shrink: 0; }
.dx-text h2 { font-size: 1.8rem; font-weight: 800; margin: 0 0 0.3rem; color: #fff; }
.dx-text p { margin: 0; color: #a0aec0; font-size: 0.92rem; line-height: 1.5; }

.conf-wrap { margin: 1.4rem 0; }
.conf-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.45rem; }
.conf-header span { font-size: 0.78rem; color: #718096; font-weight: 500; }
.conf-header strong { font-size: 1.4rem; font-weight: 800; }
.conf-track { background: #1a202c; border-radius: 999px; height: 10px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 999px; }

.clinical-card { background: #0d1117; border: 1px solid #1a202c; border-radius: 14px; padding: 1.4rem 1.6rem; margin: 0.8rem 0; }
.clinical-card .cl-title { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase; color: #4a5568; margin-bottom: 0.7rem; }
.clinical-card p { color: #a0aec0; font-size: 0.9rem; line-height: 1.65; margin: 0; }

.disclaimer { background: rgba(214,158,46,0.08); border: 1px solid rgba(214,158,46,0.25); border-radius: 10px; padding: 0.9rem 1.2rem; font-size: 0.82rem; color: #b7791f; line-height: 1.55; margin-top: 1rem; }

.live-info-box { background: rgba(229,62,62,0.07); border: 1px solid rgba(229,62,62,0.25); border-radius: 12px; padding: 1rem 1.4rem; font-size: 0.88rem; color: #fc8181; line-height: 1.6; margin: 0.8rem 0 1.2rem; }

.tab-selector { display: flex; gap: 0.6rem; margin-bottom: 1.2rem; }
.stRadio > div { flex-direction: row !important; gap: 1rem; }

.stButton > button { background: linear-gradient(135deg, #e53e3e, #c53030) !important; color: #fff !important; border: none !important; border-radius: 10px !important; padding: 0.65rem 2rem !important; font-weight: 700 !important; font-size: 0.95rem !important; width: 100% !important; letter-spacing: 0.02em !important; }
.stButton > button:hover { background: linear-gradient(135deg, #fc8181, #e53e3e) !important; }
div[data-testid="stFileUploader"] { background: #0d1117 !important; border: 1px dashed #2d3748 !important; border-radius: 12px !important; padding: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# CLASS INFO
# =====================================================================
CLASS_INFO = {
    "Normal": {
        "color": "#48bb78", "css_banner": "normal", "emoji": "💚", "short": "Normal Sinus Rhythm",
        "desc": "The heartbeat is completely normal. S1 and S2 sounds are clearly distinct, with no abnormal flow or extra beats detected.",
        "advice": "No immediate action is required. Maintain a healthy diet, regular exercise, and continue with your annual cardiac checkups.",
        "risk": "Low",
    },
    "Murmur": {
        "color": "#fc8181", "css_banner": "murmur", "emoji": "🔴", "short": "Heart Murmur Detected",
        "desc": "An abnormal blood flow sound was detected. This may indicate an underlying valvular disease or a septal defect.",
        "advice": "Please consult a cardiologist. An echocardiogram and a Doppler ultrasound are highly recommended for further diagnosis.",
        "risk": "High",
    },
    "Extrasystole": {
        "color": "#f6ad55", "css_banner": "extra", "emoji": "", "short": "Premature Heartbeat",
        "desc": "An extra or premature heartbeat was detected. The rhythm is irregular, which indicates ectopic electrical activity in the heart.",
        "advice": "Get an ECG done immediately. Avoid stress, caffeine, and alcohol. Consult a cardiologist for a thorough assessment.",
        "risk": "Medium",
    },
    "Artifact": {
        "color": "#63b3ed", "css_banner": "artifact", "emoji": "🔊", "short": "Recording Artifact",
        "desc": "External noise or movement artifacts were detected in the signal. A clean recording is required for an accurate cardiac assessment.",
        "advice": "Please record again in a quiet environment. Ensure the stethoscope is held firmly against the chest to minimize background interference.",
        "risk": "N/A",
    },
}

CLASSES = list(CLASS_INFO.keys())
RISK_COLOR = {"Low": "#48bb78", "Medium": "#f6ad55", "High": "#fc8181", "N/A": "#63b3ed"}

# =====================================================================
# LIVE AUDIO BUFFER (Thread-safe)
# =====================================================================
import threading
class AudioBuffer:
    def __init__(self):
        self.frames = []
        self.lock = threading.Lock()
        self.sample_rate = 16000
    
    def add_frame(self, frame):
        with self.lock:
            self.frames.append(frame.copy())
    
    def get_audio(self):
        with self.lock:
            if not self.frames:
                return None, self.sample_rate
            audio = np.concatenate(self.frames, axis=0)
            return audio, self.sample_rate
    
    def clear(self):
        with self.lock:
            self.frames = []

if 'audio_buffer' not in st.session_state:
    st.session_state['audio_buffer'] = AudioBuffer()

# =====================================================================
# LOAD MODEL & HELPER FUNCTIONS
# =====================================================================
@st.cache_resource
def load_ai_model():
    return load_model("heart_disease_ai_model.keras")

def load_audio_signal(audio_bytes: bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        try:
            y, sr = librosa.load(tmp_file_path, sr=22050, mono=True)
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
        
        if not np.isfinite(y).all():
            y = np.nan_to_num(y)
        return y, sr
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None, None

def extract_features_from_signal(y: np.ndarray, sr: int, n_mfcc: int = 40):
    if len(y) == 0 or len(y) < sr * 0.5:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(1, n_mfcc, 1).astype(np.float32)

# =====================================================================
# PLOT FUNCTION - FIXED for proper aspect ratio
# =====================================================================
def plot_analysis(y: np.ndarray, sr: int, view_option: str, accent: str = "#e53e3e", save_img_path=None):
    if view_option == "Both (Recommended)":
        # Wider figure for both plots
        fig = plt.figure(figsize=(14, 8), facecolor="#0d1117", dpi=120)
        gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.25)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
    else:
        fig = plt.figure(figsize=(14, 4), facecolor="#0d1117", dpi=120)
        ax1 = fig.add_subplot(111)
        ax2 = None
    
    if view_option in ["Waveform View (Time Domain)", "Both (Recommended)"]:
        ax_w = ax1
        ax_w.set_facecolor("#060910")
        times = np.linspace(0, len(y) / sr, len(y))
        ax_w.plot(times, y, color=accent, linewidth=0.7, alpha=0.9)
        ax_w.fill_between(times, y, alpha=0.12, color=accent)
        ax_w.set_ylabel("Amplitude", color="#4a5568", fontsize=8, labelpad=6)
        ax_w.tick_params(colors="#4a5568", labelsize=7)
        ax_w.set_xlim(times[0], times[-1])
        for spine in ax_w.spines.values(): spine.set_edgecolor("#1a202c")
        ax_w.text(0.01, 0.88, "TEMPORAL WAVEFORM", transform=ax_w.transAxes, fontsize=6.5, color="#4a5568", fontweight="bold")
    
    if view_option in ["Spectrogram View (Frequency Domain)", "Both (Recommended)"]:
        ax_s = ax2 if ax2 else ax1
        ax_s.set_facecolor("#060910")
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=ax_s, cmap="magma")
        ax_s.set_ylabel("Frequency (Hz)", color="#4a5568", fontsize=8, labelpad=6)
        ax_s.set_xlabel("Time (s)", color="#4a5568", fontsize=8, labelpad=4)
        ax_s.tick_params(colors="#4a5568", labelsize=7)
        for spine in ax_s.spines.values(): spine.set_edgecolor("#1a202c")
        ax_s.text(0.01, 0.88, "SPECTRAL REPRESENTATION", transform=ax_s.transAxes, fontsize=6.5, color="#4a5568", fontweight="bold")
        fig.colorbar(img, ax=ax_s, format="%+2.0f dB", pad=0.01).ax.tick_params(colors="#4a5568", labelsize=6)
    
    plt.tight_layout(pad=1.5)
    if save_img_path:
        fig.savefig(save_img_path, facecolor="#0d1117", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return fig

# =====================================================================
# PDF REPORT GENERATOR — FIXED: Single page, proper image sizing
# =====================================================================
def generate_pdf_report(patient_name, file_name, duration, prediction, confidence, info, img_path, probs, view_option):
    if FPDF is None: return None
    
    def safe_txt(text):
        return str(text).encode('latin-1', 'ignore').decode('latin-1')
    
    # ─ Color palette ──────────────────────────────────────────────────────────
    RISK_RGB = {
        "Low":    (34, 197, 94),
        "Medium": (234, 179, 8),
        "High":   (239, 68, 68),
        "N/A":    (96, 165, 250),
    }
    risk = info['risk']
    ar, ag, ab = RISK_RGB.get(risk, (229, 62, 62))
    RED   = (229, 62, 62)
    DARK  = (10, 14, 23)
    SLATE = (71, 85, 105)
    LIGHT = (248, 250, 252)
    BORDER= (226, 232, 240)
    WHITE = (255, 255, 255)
    
    bar_fill_w = max(4, int(confidence / 100 * 182))
    report_id  = f"CG-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    now_str    = datetime.now().strftime('%d %B %Y   %H:%M')
    
    pdf = FPDF()
    # Disable auto page break to force single page
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    
    # ═════════════════════════════════════════════════════════════════
    # 1.  HEADER (compact - 25mm)
    # ══════════════════════════════════════════════════════════════════
    pdf.set_fill_color(*DARK)
    pdf.rect(0, 0, 210, 25, 'F')
    
    pdf.set_fill_color(*RED)
    pdf.rect(0, 0, 210, 1.5, 'F')
    pdf.rect(0, 1.5, 4, 23.5, 'F')
    
    pdf.set_fill_color(20, 28, 42)
    pdf.rect(140, 1.5, 70, 23.5, 'F')
    
    pdf.set_fill_color(*RED)
    pdf.rect(10, 6, 10, 10, 'F')
    pdf.set_xy(10, 7)
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(*WHITE)
    pdf.cell(10, 8, safe_txt("CG"), align='C')
    
    pdf.set_xy(23, 5)
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(*WHITE)
    pdf.cell(80, 7, safe_txt("CardioGuard AI"), ln=0)
    
    pdf.set_xy(23, 13)
    pdf.set_font("Arial", '', 6)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(80, 4, safe_txt("AI-Powered Cardiac Diagnostics Report"), ln=0)
    
    pdf.set_xy(144, 6)
    pdf.set_font("Arial", '', 5)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(62, 3, safe_txt("REPORT DATE"), ln=0, align='L')
    pdf.set_xy(144, 10)
    pdf.set_font("Arial", 'B', 7)
    pdf.set_text_color(*WHITE)
    pdf.cell(62, 4, safe_txt(now_str), ln=0)
    
    pdf.set_xy(144, 16)
    pdf.set_font("Arial", '', 5)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(62, 3, safe_txt("REPORT ID"), ln=0)
    pdf.set_xy(144, 20)
    pdf.set_font("Arial", 'B', 6)
    pdf.set_text_color(*RED)
    pdf.cell(62, 3, safe_txt(report_id), ln=0)
    
    pdf.set_y(27)
    
    # ══════════════════════════════════════════════════════════════════
    # 2.  PATIENT INFORMATION CARD (compact - 18mm)
    # ══════════════════════════════════════════════════════════════════
    cy = pdf.get_y()
    
    pdf.set_fill_color(*LIGHT)
    pdf.set_draw_color(*BORDER)
    pdf.set_line_width(0.2)
    pdf.rect(10, cy, 190, 18, 'FD')
    
    pdf.set_fill_color(*RED)
    pdf.rect(10, cy, 190, 5, 'F')
    pdf.set_fill_color(180, 30, 30)
    pdf.rect(10, cy, 2.5, 5, 'F')
    
    pdf.set_xy(14, cy + 1)
    pdf.set_font("Arial", 'B', 5)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 3, safe_txt("PATIENT INFORMATION"), ln=0)
    
    fields = [
        ("PATIENT ID / NAME",    patient_name),
        ("AUDIO SOURCE",         file_name[:25] + ("..." if len(file_name) > 25 else "")),
        ("RECORDING DURATION",   f"{duration:.1f} sec"),
    ]
    col_w = 62
    for i, (lbl, val) in enumerate(fields):
        x = 12 + i * col_w
        pdf.set_xy(x, cy + 6)
        pdf.set_font("Arial", '', 4.5)
        pdf.set_text_color(*SLATE)
        pdf.cell(col_w - 2, 2.5, safe_txt(lbl), ln=0)
        pdf.set_xy(x, cy + 9)
        pdf.set_font("Arial", 'B', 6)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(col_w - 2, 3.5, safe_txt(val), ln=0)
        if i < 2:
            pdf.set_draw_color(*BORDER)
            pdf.set_line_width(0.15)
            pdf.line(10 + (i+1)*col_w, cy+5.5, 10 + (i+1)*col_w, cy+17)
    
    pdf.set_y(cy + 20)
    
    # ══════════════════════════════════════════════════════════════════
    # 3.  AI DIAGNOSIS RESULT BANNER (compact - 18mm)
    # ══════════════════════════════════════════════════════════════════
    pdf.ln(1)
    dy = pdf.get_y()
    banner_h = 18
    
    pdf.set_fill_color(ar, ag, ab)
    pdf.rect(10, dy, 190, banner_h, 'F')
    
    panel_dark = (max(ar-60,0), max(ag-60,0), max(ab-60,0))
    pdf.set_fill_color(*panel_dark)
    pdf.rect(125, dy, 75, banner_h, 'F')
    
    pdf.set_fill_color(min(ar+40,255), min(ag+40,255), min(ab+40,255))
    pdf.rect(10, dy, 190, 1.5, 'F')
    
    pdf.set_xy(14, dy + 3)
    pdf.set_font("Arial", 'B', 4)
    pdf.set_text_color(*WHITE)
    pdf.cell(100, 2.5, safe_txt("AI DIAGNOSIS RESULT"), ln=0)
    
    pdf.set_xy(14, dy + 6.5)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(*WHITE)
    pdf.cell(100, 7, safe_txt(prediction), ln=0)
    
    pdf.set_xy(14, dy + 14)
    pdf.set_font("Arial", 'I', 5.5)
    pdf.set_text_color(240, 240, 240)
    pdf.cell(100, 2.5, safe_txt(info['short']), ln=0)
    
    pdf.set_xy(129, dy + 3)
    pdf.set_font("Arial", '', 4)
    pdf.set_text_color(200, 200, 200)
    pdf.cell(35, 2.5, safe_txt("RISK LEVEL"), ln=0)
    
    pdf.set_xy(129, dy + 6.5)
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(*WHITE)
    pdf.cell(35, 5, safe_txt(risk), ln=0)
    
    pdf.set_draw_color(255, 255, 255)
    pdf.set_line_width(0.15)
    pdf.line(166, dy+3, 166, dy+16)
    
    pdf.set_xy(170, dy + 3)
    pdf.set_font("Arial", '', 4)
    pdf.set_text_color(200, 200, 200)
    pdf.cell(35, 2.5, safe_txt("AI CONFIDENCE"), ln=0)
    
    pdf.set_xy(170, dy + 6.5)
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(*WHITE)
    pdf.cell(35, 5, safe_txt(f"{confidence:.1f}%"), ln=0)
    
    pdf.set_y(dy + banner_h + 2)
    
    # ══════════════════════════════════════════════════════════════════
    # 4.  CONFIDENCE PROGRESS BAR (compact - 8mm)
    # ══════════════════════════════════════════════════════════════════
    by = pdf.get_y()
    
    pdf.set_x(10)
    pdf.set_font("Arial", 'B', 5)
    pdf.set_text_color(*SLATE)
    pdf.cell(95, 3, safe_txt("MODEL CONFIDENCE SCORE"), ln=0)
    pdf.set_font("Arial", 'B', 8)
    pdf.set_text_color(ar, ag, ab)
    pdf.cell(95, 3, safe_txt(f"{confidence:.1f}%"), ln=1, align='R')
    
    by2 = pdf.get_y()
    pdf.set_fill_color(*BORDER)
    pdf.rect(10, by2, 190, 4, 'F')
    pdf.set_fill_color(ar, ag, ab)
    pdf.rect(10, by2, bar_fill_w, 4, 'F')
    
    pdf.set_y(by2 + 5)
    
    # ══════════════════════════════════════════════════════════════════
    # 5.  CLASS PROBABILITY TABLE (compact - 25mm)
    # ══════════════════════════════════════════════════════════════════
    pdf.ln(0.5)
    tb_y = pdf.get_y()
    
    pdf.set_fill_color(*DARK)
    pdf.rect(10, tb_y, 190, 6, 'F')
    pdf.set_fill_color(*RED)
    pdf.rect(10, tb_y, 2.5, 6, 'F')
    pdf.set_xy(14, tb_y + 1.5)
    pdf.set_font("Arial", 'B', 5)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 3, safe_txt("CLASS PROBABILITY BREAKDOWN"), ln=0)
    
    tb_y += 6
    col_labels = ["Cardiac Class", "Probability", "Confidence Bar", "Status"]
    col_ws     = [45, 25, 95, 25]
    row_h      = 6
    
    CLASS_PROBS = {cls: probs[i] * 100 for i, cls in enumerate(CLASSES)}
    
    CLASS_COLORS = {
        "Normal":       (34, 197, 94),
        "Murmur":       (239, 68, 68),
        "Extrasystole": (234, 179, 8),
        "Artifact":     (96, 165, 250),
    }
    
    hx = 10
    pdf.set_fill_color(30, 41, 59)
    pdf.rect(10, tb_y, 190, row_h, 'F')
    for j, (lbl, cw) in enumerate(zip(col_labels, col_ws)):
        pdf.set_xy(hx + 2, tb_y + 1.5)
        pdf.set_font("Arial", 'B', 4.5)
        pdf.set_text_color(*WHITE)
        pdf.cell(cw - 2, 3, safe_txt(lbl), ln=0)
        hx += cw
    
    tb_y += row_h
    
    for ridx, (cls, prob) in enumerate(CLASS_PROBS.items()): 
        row_bg = (252, 252, 253) if ridx % 2 == 0 else (244, 246, 250)
        pdf.set_fill_color(*row_bg)
        pdf.set_draw_color(*BORDER)
        pdf.set_line_width(0.1)
        pdf.rect(10, tb_y, 190, row_h, 'FD')
        
        if cls == prediction:
            pdf.set_fill_color(min(ar,255), min(ag,255), min(ab,255))
            pdf.set_draw_color(ar, ag, ab)
            pdf.rect(10, tb_y, 190, row_h, 'FD')
            txt_color = WHITE
        else:
            txt_color = (15, 23, 42)
        
        hx = 10
        cr, cg, cb = CLASS_COLORS[cls]
        
        pdf.set_fill_color(cr, cg, cb)
        pdf.rect(hx + 2, tb_y + 2, 2, 2, 'F')
        pdf.set_xy(hx + 5.5, tb_y + 1.5)
        pdf.set_font("Arial", 'B', 6)
        pdf.set_text_color(*txt_color)
        pdf.cell(col_ws[0] - 5.5, 3, safe_txt(cls), ln=0)
        hx += col_ws[0]
        
        pdf.set_xy(hx + 2, tb_y + 1.5)
        pdf.set_font("Arial", 'B', 6)
        pdf.set_text_color(*txt_color)
        pdf.cell(col_ws[1] - 2, 3, safe_txt(f"{prob:.1f}%"), ln=0)
        hx += col_ws[1]
        
        bar_x  = hx + 2
        bar_bw = col_ws[2] - 6
        bar_bh = 2.5
        bar_by = tb_y + 1.75
        pdf.set_fill_color(220, 225, 232)
        pdf.rect(bar_x, bar_by, bar_bw, bar_bh, 'F')
        filled = max(1, int(prob / 100 * bar_bw))
        pdf.set_fill_color(cr, cg, cb)
        pdf.rect(bar_x, bar_by, filled, bar_bh, 'F')
        hx += col_ws[2]
        
        pdf.set_xy(hx + 2, tb_y + 1.5)
        if cls == prediction:
            pdf.set_fill_color(*WHITE)
            pdf.rect(hx + 2, tb_y + 1.5, 16, 3.5, 'F')
            pdf.set_xy(hx + 2, tb_y + 1.75)
            pdf.set_font("Arial", 'B', 4.5)
            pdf.set_text_color(ar, ag, ab)
            pdf.cell(16, 2.5, safe_txt("DETECTED"), align='C', ln=0)
        else:
            pdf.set_font("Arial", '', 5)
            pdf.set_text_color(*SLATE)
            pdf.set_xy(hx + 2, tb_y + 1.5)
            pdf.cell(16, 2.5, safe_txt("--"), ln=0)
        
        tb_y += row_h
    
    pdf.set_y(tb_y + 2)
    
    # ══════════════════════════════════════════════════════════════════
    # 6.  TWO-COLUMN CLINICAL CARDS (compact - 22mm)
    # ══════════════════════════════════════════════════════════════════
    pdf.ln(0.5)
    cl_y   = pdf.get_y()
    card_h = 22
    
    for i, (title, body) in enumerate([
        ("CLINICAL FINDINGS",  info['desc']),
        ("RECOMMENDED ACTION", info['advice']),
    ]):
        cx = 10 + i * 96
        pdf.set_fill_color(*LIGHT)
        pdf.set_draw_color(*BORDER)
        pdf.set_line_width(0.2)
        pdf.rect(cx, cl_y, 92, card_h, 'FD')
        pdf.set_fill_color(ar, ag, ab)
        pdf.rect(cx, cl_y, 92, 5, 'F')
        pdf.set_fill_color(max(ar-40,0), max(ag-40,0), max(ab-40,0))
        pdf.rect(cx, cl_y, 2.5, 5, 'F')
        pdf.set_xy(cx + 4, cl_y + 1)
        pdf.set_font("Arial", 'B', 4.5)
        pdf.set_text_color(*WHITE)
        pdf.cell(85, 3, safe_txt(title), ln=0)
        pdf.set_xy(cx + 3, cl_y + 6)
        pdf.set_font("Arial", '', 5.5)
        pdf.set_text_color(51, 65, 85)
        pdf.multi_cell(86, 3, safe_txt(body))
    
    pdf.set_y(cl_y + card_h + 1.5)
    
    # ══════════════════════════════════════════════════════════════════
    # 7.  SIGNAL VISUALIZATION - FIXED: Dynamic height, no cutoff
    # ══════════════════════════════════════════════════════════════════
    if img_path and os.path.exists(img_path):
        try:
            from PIL import Image as PILImage
            img = PILImage.open(img_path)
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                bg = PILImage.new('RGB', img.size, (10, 14, 23))
                bg.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                safe_img_path = img_path.replace(".png", "_safe.jpg")
                bg.save(safe_img_path, "JPEG", quality=93)
                img_path = safe_img_path
            else:
                safe_img_path = img_path.replace(".png", "_rgb.jpg")
                img.convert('RGB').save(safe_img_path, "JPEG", quality=93)
                img_path = safe_img_path
            
            viz_y = pdf.get_y()
            pdf.set_fill_color(*DARK)
            pdf.rect(10, viz_y, 190, 6, 'F')
            pdf.set_fill_color(*RED)
            pdf.rect(10, viz_y, 2.5, 6, 'F')
            pdf.set_xy(14, viz_y + 1.5)
            pdf.set_font("Arial", 'B', 5)
            pdf.set_text_color(*WHITE)
            pdf.cell(180, 3, safe_txt("SIGNAL VISUALIZATION  —  Temporal Waveform & Spectral Representation"), ln=0)
            pdf.set_y(viz_y + 6)
            
            # ── FIX: Calculate proper image height based on view_option ──
            img_y = pdf.get_y()
            
            # Get image pixel dimensions
            img_w_px, img_h_px = img.size
            aspect_ratio = img_h_px / float(img_w_px)
            
            # Full page width
            pdf_img_w = 190
            
            # Calculate proportional height
            pdf_img_h = pdf_img_w * aspect_ratio
            
            # FIX: Different max heights for single vs both visualizations
            if view_option == "Both (Recommended)":
                max_img_h = 55  # Both plots need more space
            else:
                max_img_h = 28  # Single plot needs less space
            
            # Cap height if too large
            if pdf_img_h > max_img_h:
                pdf_img_h = max_img_h
            
            # Add background and image
            pdf.set_fill_color(*DARK)
            pdf.rect(10, img_y, 190, pdf_img_h + 1, 'F')
            pdf.image(img_path, x=10, y=img_y + 0.5, w=pdf_img_w, h=pdf_img_h)
            pdf.set_y(img_y + pdf_img_h + 1.5)
        except Exception:
            pass
    
    # ══════════════════════════════════════════════════════════════════
    # 8.  MEDICAL DISCLAIMER (at bottom of page)
    # ══════════════════════════════════════════════════════════════════
    # Position disclaimer at fixed position near bottom
    dsc_y = 260
    
    pdf.set_fill_color(255, 251, 235)
    pdf.set_draw_color(217, 166, 28)
    pdf.set_line_width(0.3)
    pdf.rect(10, dsc_y, 190, 15, 'FD')
    
    pdf.set_fill_color(217, 166, 28)
    pdf.rect(10, dsc_y, 3, 15, 'F')
    
    pdf.set_xy(16, dsc_y + 2)
    pdf.set_font("Arial", 'B', 5)
    pdf.set_text_color(133, 95, 0)
    pdf.cell(0, 2.5, safe_txt("! MEDICAL DISCLAIMER"), ln=0)
    
    pdf.set_xy(16, dsc_y + 5.5)
    pdf.set_font("Arial", '', 5)
    pdf.set_text_color(107, 78, 0)
    pdf.multi_cell(180, 2.5, safe_txt(
        "This report is generated by an AI research tool intended for educational and preliminary screening purposes ONLY. "
        "It does NOT constitute a definitive medical diagnosis. All results must be reviewed and confirmed by a licensed "
        "cardiologist before any clinical decision is made."
    ))
    
    # ══════════════════════════════════════════════════════════════════
    # 9.  FOOTER BAR
    # ═════════════════════════════════════════════════════════════════
    footer_y = 280
    pdf.set_fill_color(*DARK)
    pdf.rect(0, footer_y, 210, 10, 'F')
    pdf.set_fill_color(*RED)
    pdf.rect(0, footer_y, 210, 1.5, 'F')
    
    pdf.set_xy(10, footer_y + 2.5)
    pdf.set_font("Arial", '', 5)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(95, 3, safe_txt("CardioGuard AI | AI-Powered Cardiac Analysis Platform"), ln=0)
    pdf.cell(95, 3, safe_txt(f"{report_id} | Page 1 of 1"), ln=0, align='R')
    
    # Save
    safe_filename = safe_txt(patient_name).replace(' ', '_').replace('/', '-')
    report_path   = os.path.join("reports", f"CardioGuard_Report_{safe_filename}.pdf")
    pdf.output(report_path)
    return report_path

# =====================================================================
# SHARED ANALYSIS UI (used by both upload & live)
# =====================================================================
def run_analysis_ui(y_signal, sr, source_name, source_label, model):
    """Visualization + AI diagnosis + PDF — shared between Upload and Live modes."""
    unique_id = uuid.uuid4().hex[:6]
    temp_img_path = os.path.join("recordings", f"plot_{unique_id}.png")
    
    total_duration = len(y_signal) / sr
    
    # ── Trim slider ──────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Audio Trimming & Range Selection</div>', unsafe_allow_html=True)
    
    default_end = min(float(total_duration), 10.0)
    
    start_time, end_time = st.slider(
        "Select Audio Range (Select the best 5–10 seconds of relevant audio)",
        min_value=0.0,
        max_value=float(total_duration),
        value=(0.0, default_end),
        step=0.1,
        format="%.1fs",
        key=f"trim_slider_{source_label}"
    )
    
    y_work = y_signal[int(start_time * sr):int(end_time * sr)]
    duration_used = end_time - start_time
    
    # Playable trimmed audio
    trimmed_io = io.BytesIO()
    y_int16 = np.clip(y_work * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(trimmed_io, sr, y_int16)
    st.audio(trimmed_io.getvalue(), format="audio/wav")
    
    # ── Visualization ──────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Step 02 — Signal Analysis</div>', unsafe_allow_html=True)
    st.subheader(" Interactive Visualizations")
    
    visual_option = st.radio(
        "Select Visualization Type:",
        ["Waveform View (Time Domain)", "Spectrogram View (Frequency Domain)", "Both (Recommended)"],
        horizontal=True,
        key=f"vis_{source_label}"
    )
    
    with st.spinner("Generating Visualization..."):
        fig = plot_analysis(y_work, sr, visual_option, save_img_path=temp_img_path)
        st.pyplot(fig)
        plt.close(fig)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # ── Analyze button ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Step 03 — AI Diagnosis</div>', unsafe_allow_html=True)
    col_btn, col_blank = st.columns([1, 2])
    with col_btn:
        analyze_clicked = st.button("🔍 Run Cardiac Analysis", key=f"analyze_{source_label}")
    
    pred_key = f"pred_done_{source_label}"
    
    if analyze_clicked:
        if model is None:
            st.error("Model failed to load.")
            return
        with st.spinner("Deep learning model analyzing..."):
            features = extract_features_from_signal(y_work, sr)
            if features is None:
                st.error("Feature extraction failed. Please provide a valid audio segment (min 0.5s).")
                return
            predicted_class, confidence, probs = predict_heartbeat(model, features)
            if predicted_class == "Model Failed":
                st.error("Prediction failed.")
                return
        
        st.session_state[pred_key] = True
        st.session_state[f'predicted_class_{source_label}'] = predicted_class
        st.session_state[f'confidence_{source_label}'] = confidence
        st.session_state[f'probs_{source_label}'] = probs
        st.session_state[f'duration_{source_label}'] = duration_used
        st.session_state[f'filename_{source_label}'] = source_name
        st.session_state.pop(f'pdf_bytes_{source_label}', None)
    
    if st.session_state.get(pred_key, False):
        predicted_class = st.session_state[f'predicted_class_{source_label}']
        confidence      = st.session_state[f'confidence_{source_label}']
        probs           = st.session_state[f'probs_{source_label}']
        info  = CLASS_INFO[predicted_class]
        color = info["color"]
        risk  = info["risk"]
        
        st.markdown(f"""
        <div class="dx-banner {info['css_banner']}">
            <div class="dx-icon">{info['emoji']}</div>
            <div class="dx-text">
                <h2>{predicted_class}</h2>
                <p>{info['short']}  &nbsp;· &nbsp; Risk Level: <strong style="color:{RISK_COLOR[risk]}">{risk}</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="conf-wrap">
            <div class="conf-header">
                <span>MODEL CONFIDENCE</span>
                <strong style="color:{color};">{confidence:.1f}%</strong>
            </div>
            <div class="conf-track">
                <div class="conf-fill" style="width:{confidence:.1f}%; background:{color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-label" style="margin-top:1.4rem;">📊 Detailed Probability Split Chart</div>', unsafe_allow_html=True)
        chart_data = pd.DataFrame({
            'Cardiac Classes': CLASSES,
            'Probability (%)': [p * 100 for p in probs]
        }).set_index('Cardiac Classes')
        st.bar_chart(data=chart_data, color="#e53e3e")
        
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Clinical Findings & Recommendation</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="clinical-card">
                <div class="cl-title">🔬 Findings</div>
                <p>{info['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="clinical-card">
                <div class="cl-title">💊 Recommended Action</div>
                <p>{info['advice']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # ── PDF Report ────────────────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Generate PDF Medical Report</div>', unsafe_allow_html=True)
        
        with st.form(f"pdf_form_{source_label}"):
            patient_name = st.text_input("Enter Patient ID / Name", placeholder="e.g. John Doe / Pt-1029")
            submitted = st.form_submit_button("Generate PDF Report")
            if submitted:
                if FPDF is None:
                    st.error("FPDF library missing.")
                else:
                    pt_name = patient_name if patient_name else "Anonymous"
                    with st.spinner("Generating PDF..."):
                        report_path = generate_pdf_report(
                            pt_name,
                            st.session_state[f'filename_{source_label}'],
                            st.session_state[f'duration_{source_label}'],
                            predicted_class, confidence, info, temp_img_path, probs, visual_option
                        )
                        with open(report_path, "rb") as f:
                            PDFbyte = f.read()
                    st.session_state[f'pdf_bytes_{source_label}'] = PDFbyte
                    st.session_state[f'pdf_filename_{source_label}'] = os.path.basename(report_path)
                    st.success(f"✅ Report successfully generated for {pt_name}!")
        
        # Download button FORM ke bahar
        if st.session_state.get(f'pdf_bytes_{source_label}'):
            st.download_button(
                label="📥 Download PDF Report",
                data=st.session_state[f'pdf_bytes_{source_label}'],
                file_name=st.session_state.get(f'pdf_filename_{source_label}', 'CardioGuard_Report.pdf'),
                mime='application/octet-stream',
                key=f"dl_{source_label}"
            )
        
        st.markdown("""
        <div class="disclaimer">
            ️  <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only.
            Please consult a licensed cardiologist for any medical diagnosis. AI results do not constitute a final medical opinion.
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# MAIN UI
# =====================================================================
def main():
    st.sidebar.title("❤️ Navigation Menu")
    menu = st.sidebar.radio("Go To:", ["Upload & Diagnostics", "Model Insights / Analytics"])
    model = load_ai_model()
    
    # ── Model Insights ────────────────────────────────────────────────────────
    if menu == "Model Insights / Analytics":
        st.title(" Model Insights & Performance Analytics")
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.subheader("Training Architecture Parameters")
        st.code("""
Layer (type)                Output Shape              Param #
conv1d (Conv1D)             (None, 40, 32)            128
max_pooling1d               (None, 20, 32)            0
conv1d_1 (Conv1D)           (None, 20, 64)            6208
max_pooling1d_1             (None, 10, 64)            0
flatten (Flatten)           (None, 640)               0
dense (Dense)               (None, 128)               82048
dropout (Dropout)           (None, 128)               0
dense_1 (Dense)             (None, 4)                 516
Total params: 88,900
        """)
        st.subheader("Validation Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Accuracy", "94.2%", "+1.5%")
        m2.metric("Precision (Murmur)", "0.92")
        m3.metric("Recall (Normal)", "0.96")
        st.info("The system handles background noise and classifies standard PCG audio into Normal, Murmur, Extrasystole, and Artifacts.")
        return
    
    # ── Upload & Diagnostics ──────────────────────────────────────────────────
    st.markdown("""
    <div class="header-wrap">
        <div class="badge">AI-Powered Cardiac Analysis</div>
        <h1>❤️ CardioGuard AI</h1>
        <p>Clinical-grade Heart Disease Detection via Heartbeat Sound Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model is not None:
        st.success("✅ AI Model Loaded Successfully")
    else:
        st.error("❌ Model failed to load. Please check the terminal for errors.")
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # ── INPUT MODE TABS ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Step 01 — Choose Audio Input Method</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Audio Input Method:",
        ["📁 Upload Audio File", "🎙️ Live Microphone Recording"],
        horizontal=True,
        key="input_mode"
    )
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODE 1 — UPLOAD
    # ═════════════════════════════════════════════════════════════════════════
    if input_mode == "📁 Upload Audio File":
        st.subheader("🎙️ Upload Heartbeat Recording")
        uploaded_file = st.file_uploader(
            "Select a WAV or MP3 file",
            type=["wav", "mp3"],
            help="200MB per file • WAV / MP3"
        )
        
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id  = uuid.uuid4().hex[:6]
            file_ext   = os.path.splitext(uploaded_file.name)[1]
            save_path  = os.path.join("recordings", f"{timestamp}_{unique_id}{file_ext}")
            with open(save_path, "wb") as f:
                f.write(audio_bytes)
            
            y_full, sr_full = load_audio_signal(audio_bytes)
            if y_full is not None:
                run_analysis_ui(y_full, sr_full, uploaded_file.name, "upload", model)
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODE 2 — LIVE RECORDING
    # ══════════════════════════════════════════════════════════════════════════
    else:
        st.subheader("🎙️ Live Heartbeat Recording")
        
        if not WEBRTC_AVAILABLE:
            st.error("streamlit-webrtc not installed. Run: `pip install streamlit-webrtc av`")
            return
        
        st.markdown("""
        <div class="live-info-box">
            📋  <strong>How to record:</strong> <br>
            1. Click <strong>START</strong> below and allow microphone access <br>
            2. Place your microphone / digital stethoscope on your chest <br>
            3. Stay still and breathe normally for <strong>10–15 seconds</strong> <br>
            4. Click <strong>STOP</strong> when done <br>
            5. Click <strong>"Use This Recording"</strong> to analyze
        </div>
        """, unsafe_allow_html=True)
        
        audio_buffer = st.session_state['audio_buffer']
        
        # WebRTC audio callback
        def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
            pcm = frame.to_ndarray()
            mono = pcm.mean(axis=0)
            audio_buffer.add_frame(mono.astype(np.float32) / 32768.0)
            return frame
        
        RTC_CONFIG = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        ctx = webrtc_streamer(
            key="live_heartbeat",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=RTC_CONFIG,
            audio_frame_callback=audio_frame_callback,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
        )
        
        # Recording state tracking
        if ctx.state.playing:
            st.info("🔴 Recording in progress... Speak/place stethoscope now.")
            if st.session_state.get('was_recording') == False:
                audio_buffer.clear()
            st.session_state['was_recording'] = True
        else:
            if st.session_state.get('was_recording', False):
                st.session_state['was_recording'] = False
                y_live, sr_live = audio_buffer.get_audio()
                if y_live is not None and len(y_live) > sr_live * 1:
                    st.session_state['live_audio'] = y_live
                    st.session_state['live_sr'] = sr_live
                    st.success(f"✅ Recording captured! Duration: {len(y_live)/sr_live:.1f}s")
            st.session_state['was_recording'] = st.session_state.get('was_recording', False)
        
        # Show recorded audio & analysis
        if st.session_state.get('live_audio') is not None:
            y_live = st.session_state['live_audio']
            sr_live = st.session_state['live_sr']
            duration_live = len(y_live) / sr_live
            
            st.markdown(f"**Recorded:** {duration_live:.1f} seconds")
            
            # Playback
            live_io = io.BytesIO()
            y_int16 = np.clip(y_live * 32767, -32768, 32767).astype(np.int16)
            wavfile.write(live_io, sr_live, y_int16)
            st.audio(live_io.getvalue(), format="audio/wav")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                use_recording = st.button("✅ Use This Recording & Analyze", key="use_live")
            with col2:
                if st.button("🔄 Record Again", key="re_record"):
                    audio_buffer.clear()
                    st.session_state.pop('live_audio', None)
                    st.session_state.pop('live_sr', None)
                    st.session_state.pop('pred_done_live', None)
                    st.session_state.pop('pdf_bytes_live', None)
                    st.rerun()
            
            if use_recording or st.session_state.get('live_analysis_ready', False):
                if use_recording:
                    st.session_state['live_analysis_ready'] = True
                    y_resampled = librosa.resample(y_live, orig_sr=sr_live, target_sr=22050)
                    st.session_state['live_audio_22k'] = y_resampled
                
                y_analysis = st.session_state.get('live_audio_22k', y_live)
                run_analysis_ui(y_analysis, 22050, "Live Microphone Recording", "live", model)
    
    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Methodology</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    steps = [
        ("01", "Audio Acquisition", "Upload a recorded heartbeat in WAV or MP3 format, or record live using your microphone or digital stethoscope."),
        ("02", "Feature Extraction", "MFCC (Mel-frequency cepstral coefficients) are extracted using Librosa, followed by temporal-spectral analysis."),
        ("03", "Deep Learning Classification", "The trained CNN model classifies the signal into 4 cardiac classes along with a confidence score."),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3], steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">Step {num}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        CardioGuard AI  &nbsp;· &nbsp; For Research & Educational Purposes Only
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()