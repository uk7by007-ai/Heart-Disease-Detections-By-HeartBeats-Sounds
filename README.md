# CardioGuard AI - Heart Disease Detection ❤️

CardioGuard AI is an advanced, AI-powered web application built with Streamlit that analyzes heartbeat sounds (phonocardiograms) to detect potential heart diseases. By leveraging deep learning and digital signal processing techniques, it can classify heartbeat audio recordings into four distinct categories and provide detailed diagnostic reports.

## Key Features

### 🎧 Comprehensive Audio Input
- **File Upload**: Easily upload existing `.wav` recordings from digital stethoscopes or other medical devices.
- **Live Recording via WebRTC**: Record heartbeats in real-time directly through your browser securely and conveniently.

### ✂️ Precision Signal Tuning
- **Interactive Audio Trimming**: Use intuitive sliders to isolate the most relevant 5-10 seconds of audio, eliminating unneeded silence or initial handling noise for improved model accuracy.

### 📊 Advanced Interactive Visualizations
- **Temporal Waveforms**: Observe the raw amplitude of the heartbeat over time in a sleek, high-contrast dark mode visualization.
- **Spectral Representations (Spectrograms)**: Visualize the frequency distribution of the audio signal to identify subtle murmurs or anomalies that are hard to hear.

### 🧠 Deep Learning AI Diagnosis
- **Robust Classification**: Powered by a pre-trained Keras neural network that extracts Mel-Frequency Cepstral Coefficients (MFCCs) to classify sounds.
- **4 Distinct Diagnostic Classes**:
  - **Normal**: Normal Sinus Rhythm (S1 and S2 sounds distinct).
  - **Murmur**: Abnormal blood flow sounds indicating possible valvular or septal defects.
  - **Extrasystole**: Premature or irregular heartbeat detections.
  - **Artifact**: External noise/movement interference prompting a re-recording.
- **Confidence Scoring**: Displays exact AI model confidence percentages alongside a visual probability breakdown for all classes.

### 📄 Clinical-Grade PDF Reports
- **One-Click Export**: Generate professional, single-page PDF reports summarizing the diagnosis.
- **Detailed Insights**: Includes patient details, visual graphs, probability tables, risk level indicators, clinical findings, and recommended actions.

### 🎨 Premium UI/UX Design
- **Modern Interface**: A stunning, responsive, dark-themed user interface utilizing custom CSS, glassmorphism hints, and modern typography (Inter).
- **Clear Visual Hierarchy**: Color-coded diagnosis banners and risk indicators (Low, Medium, High) make interpreting results immediate and intuitive.

## Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Audio Processing**: [Librosa](https://librosa.org/), SciPy
- **Machine Learning**: TensorFlow / Keras (`heart_disease_ai_model.keras`)
- **Visualizations**: Matplotlib
- **PDF Generation**: FPDF
- **Live Audio**: Streamlit-WebRTC

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd "Ai Heart Disease Detections By Heartbeats Sounds"
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: You may need to install `ffmpeg` or `libsndfile` on your system for `librosa` and `av` to work properly).*

## Usage

1. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

2. **Navigate the app:**
   - Open your browser to `http://localhost:8501`.
   - Choose between **Upload Audio** or **Live Recording** from the sidebar.
   - Follow the on-screen instructions to analyze the heartbeat sound.
   - Generate and download your PDF report.

## Directory Structure

```
├── app.py                            # Main Streamlit application
├── heart_disease_ai_model.keras      # Pre-trained deep learning model
├── requirements.txt                  # Python dependencies
├── utils/
│   ├── audio_processing.py           # Audio processing utility functions
│   ├── prediction.py                 # Model loading and prediction logic
│   └── visualization.py              # Visualization helpers
├── recordings/                       # Temporary storage for audio and plots
└── reports/                          # Generated PDF reports
```

## Disclaimer

**! MEDICAL DISCLAIMER**
This application is an AI research tool intended for educational and preliminary screening purposes ONLY. It does NOT constitute a definitive medical diagnosis. All results must be reviewed and confirmed by a licensed cardiologist before any clinical decision is made.
