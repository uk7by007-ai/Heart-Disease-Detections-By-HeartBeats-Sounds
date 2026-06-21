import numpy as np
import librosa
import io
import os
import soundfile as sf
from scipy.signal import butter, lfilter

# =====================================================================
# CONSTANTS
# =====================================================================
SAMPLE_RATE   = 22050   # Hz — librosa default
N_MFCC        = 40      # MFCC features count (model input size)
MIN_DURATION  = 0.5     # seconds — is se chhota audio reject

# Noise Removal Constants (Heartbeat Frequency Range: 20Hz - 500Hz)
LOWCUT        = 20.0    # Hz
HIGHCUT       = 500.0   # Hz


# =====================================================================
# NOISE REMOVAL FILTER (Butterworth Bandpass)
# =====================================================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Background noise remove karne ke liye Bandpass Filter apply karta hai.
    """
    try:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bar')
        y = lfilter(b, a, data)
        return y.astype(np.float32)
    except Exception as e:
        print(f"[WARNING] Filter apply nahi ho saka, raw audio use ho raha hai: {e}")
        return data


# =====================================================================
# LOAD AUDIO
# =====================================================================
def load_audio(source) -> tuple[np.ndarray, int] | tuple[None, None]:
    """
    Audio load karo — file path ya bytes dono accept karta hai.

    Args:
        source: str (file path) ya bytes (file content)

    Returns:
        (y, sr) numpy array + sample rate, ya (None, None) on error
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            y, sr = librosa.load(io.BytesIO(source), sr=SAMPLE_RATE, mono=True)
        elif isinstance(source, str):
            if not os.path.exists(source):
                print(f"[ERROR] File nahi mila: {source}")
                return None, None
            y, sr = librosa.load(source, sr=SAMPLE_RATE, mono=True)
        else:
            print("[ERROR] Source sirf file path ya bytes hona chahiye.")
            return None, None

        return y, sr

    except Exception as e:
        print(f"[ERROR] Audio load fail: {e}")
        return None, None


# =====================================================================
# VALIDATE AUDIO
# =====================================================================
def validate_audio(y: np.ndarray, sr: int) -> tuple[bool, str]:
    """
    Audio ko validate karo — length aur silence check.

    Returns:
        (is_valid: bool, message: str)
    """
    if y is None or len(y) == 0:
        return False, "Audio empty hai."

    duration = len(y) / sr
    if duration < MIN_DURATION:
        return False, f"Audio bahut chhota hai ({duration:.2f}s). Kam az kam {MIN_DURATION}s chahiye."

    # Silence check — agar sab zeros hain
    if np.max(np.abs(y)) < 1e-6:
        return False, "Audio bilkul silent hai. Sahi recording use karein."

    return True, f"Audio valid hai ({duration:.2f}s @ {sr}Hz)"


# =====================================================================
# EXTRACT MFCC FEATURES
# =====================================================================
def extract_mfcc(y: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray | None:
    """
    MFCC features extract karo audio se.

    Args:
        y     : audio waveform
        sr    : sample rate
        n_mfcc: number of MFCC features (default 40)

    Returns:
        numpy array shape (1, n_mfcc, 1) — model ready input
    """
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_scaled = np.mean(mfcc.T, axis=0)          # shape: (n_mfcc,)
        features = mfcc_scaled.reshape(1, n_mfcc, 1)   # shape: (1, 40, 1)
        return features.astype(np.float32)
    except Exception as e:
        print(f"[ERROR] MFCC extraction fail: {e}")
        return None


# =====================================================================
# EXTRACT ADDITIONAL FEATURES (optional — for richer models)
# =====================================================================
def extract_all_features(y: np.ndarray, sr: int) -> dict | None:
    """
    Agar model additional features use karta ho toh yeh use karo.
    Returns dictionary with all audio features.
    """
    try:
        features = {}

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        features["mfcc"] = np.mean(mfcc.T, axis=0)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features["chroma"] = np.mean(chroma.T, axis=0)

        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features["spectral_contrast"] = np.mean(contrast.T, axis=0)

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features["zcr"] = np.mean(zcr)

        # RMS Energy
        rms = librosa.feature.rms(y=y)
        features["rms"] = np.mean(rms)

        return features

    except Exception as e:
        print(f"[ERROR] Feature extraction fail: {e}")
        return None


# =====================================================================
# MAIN PIPELINE — source se seedha model-ready features
# =====================================================================
def process_audio(source) -> np.ndarray | None:
    """
    Complete pipeline:
      source (bytes/path) → load → validate → Noise Filter → MFCC → (1, 40, 1)

    Yeh function app.py aur prediction.py dono use karte hain.

    Args:
        source: audio bytes ya file path

    Returns:
        numpy array (1, 40, 1) ya None on failure
    """
    # Step 1 — Load
    y, sr = load_audio(source)
    if y is None:
        return None

    # Step 2 — Validate
    is_valid, message = validate_audio(y, sr)
    if not is_valid:
        print(f"[WARNING] {message}")
        return None

    print(f"[INFO] {message}")

    # Step 2.5 — Apply Noise Filter (Bina baaki code chhede noise removal process)
    print("[INFO] Applying Noise Removal Filter (20Hz - 500Hz Bandpass)...")
    y = butter_bandpass_filter(y, LOWCUT, HIGHCUT, sr)

    # Step 3 — Extract features (Cleaned audio se features niklenge)
    features = extract_mfcc(y, sr)
    if features is None:
        return None

    print(f"[INFO] Features extracted from cleaned audio: shape={features.shape}, dtype={features.dtype}")
    return features


# =====================================================================
# KEY WAVEFORM DATA (Cleaned for perfect graph visualization)
# =====================================================================
def get_waveform_data(source) -> tuple[np.ndarray, int] | tuple[None, None]:
    """
    Waveform plot ke liye noise-filtered audio data return karo.
    Is se web page par waveform graph bhi bilkul saaf dikhega.
    """
    y, sr = load_audio(source)
    if y is not None:
        y = butter_bandpass_filter(y, LOWCUT, HIGHCUT, sr)
    return y, sr


# =====================================================================
# DIRECT RUN (testing ke liye)
# =====================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_processing.py <audio_file.wav>")
        print("\nDummy test chal raha hai...")

        # Dummy sine wave test
        sr = SAMPLE_RATE
        t  = np.linspace(0, 2, sr * 2)
        y  = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        is_valid, msg = validate_audio(y, sr)
        print(f"Validation: {msg}")

        # Test Filter
        y_clean = butter_bandpass_filter(y, LOWCUT, HIGHCUT, sr)

        features = extract_mfcc(y_clean, sr)
        if features is not None:
            print(f"Features shape: {features.shape}")
            print(f"Features dtype: {features.dtype}")
            print(f"Min: {features.min():.4f} | Max: {features.max():.4f}")
        else:
            print("Feature extraction fail.")
    else:
        audio_file = sys.argv[1]
        print(f"Processing: {audio_file}")
        features = process_audio(audio_file)
        if features is not None:
            print(f"✅ Features ready: {features.shape}")
        else:
            print("❌ Processing fail.")