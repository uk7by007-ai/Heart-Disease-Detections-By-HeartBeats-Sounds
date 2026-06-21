import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def show_waveform(audio_path):
    audio, sr = librosa.load(audio_path)

    fig, ax = plt.subplots(figsize=(10, 3))

    librosa.display.waveshow(audio, sr=sr, ax=ax)

    ax.set_title("Heartbeat Waveform")

    st.pyplot(fig)

def show_spectrogram(audio_path):
    # Audio signal load karein
    y, sr = librosa.load(audio_path, sr=None)
    
    # Short-time Fourier transform (STFT)
    stft = librosa.stft(y)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    
    # Spectrogram Plot Generate karein
    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='linear', ax=ax, cmap='viridis')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    
    ax.set_title("Spectrogram Visualization (Frequency Domain)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    
    st.pyplot(fig)