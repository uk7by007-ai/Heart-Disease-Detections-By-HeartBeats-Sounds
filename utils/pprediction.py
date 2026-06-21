from tensorflow.keras.models import load_model
import streamlit as st

@st.cache_resource
def load_ai_model():
    model = load_model("heart_disease_ai_model.keras")
    return model

classes = [
    "Normal",
    "Murmur",
    "Extrasystole",
    "Artifact"
]

def predict_heartbeat(model, features):
    prediction = model.predict(features)

    predicted_index = prediction.argmax()

    predicted_class = classes[predicted_index]

    confidence = float(prediction.max()) * 100

    return predicted_class, confidence