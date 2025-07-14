import streamlit as st
import cv2
import numpy as np
from utils.predictor import load_model, predict_image
from utils.tts import speak
from PIL import Image
import tempfile
import os

# Load model and classes
model = load_model("Saved_Models/model_07_13_2025_22_12_53.h5")
classes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'nothing', 'space'
]

st.title("🤟 Sign Language Translator with TTS & Voice")
st.sidebar.header("Upload Section")

# === Image Upload ===
img_file = st.sidebar.file_uploader("📷 Upload an Image", type=['jpg', 'png', 'jpeg'])
if img_file:
    image = Image.open(img_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_array = np.array(image)
    label, confidence = predict_image(model, img_array, classes)
    st.success(f"Prediction: {label} ({confidence * 100:.2f}%)")
    if st.button("🔊 Speak Prediction"):
        speak(label)

# === Video Upload ===
video_file = st.sidebar.file_uploader("📹 Upload a Video", type=["mp4", "avi"])
if video_file:
    st.video(video_file)
    temp_vid = tempfile.NamedTemporaryFile(delete=False)
    temp_vid.write(video_file.read())

    cap = cv2.VideoCapture(temp_vid.name)
    predicted_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        label, conf = predict_image(model, frame, classes)
        if conf > 0.75:
            predicted_sequence.append(label)

    cap.release()
    sentence = "".join(predicted_sequence)
    st.success(f"Predicted Sentence: {sentence}")
    if st.button("🔊 Speak Sentence"):
        speak(sentence)

# === Live Webcam ===
if st.checkbox("📷 Use Webcam"):
    st.warning("Webcam support requires Streamlit WebRTC or local OpenCV loop.")
    st.markdown("Run separately in `realtime_predict.py`")

# === Voice-to-Text ===
if st.button("🎙️ Speak to Text"):
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        st.info("🎤 Listening...")
        audio = recognizer.listen(source, timeout=5)
        try:
            transcript = recognizer.recognize_google(audio)
            st.success(f"📝 You said: {transcript}")
        except:
            st.error("❌ Couldn't recognize your voice. Try again.")
