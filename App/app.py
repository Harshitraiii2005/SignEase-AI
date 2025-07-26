from flask import Flask, render_template, request, jsonify
from utilss.mediapipe_predictor import predict_from_image
from utilss.tts import speak_from_text
from prometheus_flask_exporter import PrometheusMetrics  # 🔥 NEW
import os, cv2, tempfile
from PIL import Image
import speech_recognition as sr

app = Flask(__name__)

# === Prometheus Metrics Setup ===
metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Signease AI App Metrics', version='1.0.0')

# === Custom Metrics ===
from prometheus_client import Counter

total_users = Counter('signease_total_users', 'Total users who visited')
request_count = Counter('signease_total_requests', 'Total requests made')
service_usage = Counter('signease_service_usage', 'Service usage count', ['service'])

@app.route('/')
def index():
    total_users.inc()  # 👤 Increment user count
    return render_template('index.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/app', methods=['GET', 'POST'])
def app_page():
    prediction = None
    sentence = None
    audio_file = None

    if request.method == 'POST':
        request_count.inc()  # 📊 Count request

        # === IMAGE PREDICTION ===
        if 'image_file' in request.files:
            img_file = request.files['image_file']
            if img_file.filename != '':
                prediction, confidence = predict_from_image(img_file)
                audio_file = speak_from_text(prediction)
                service_usage.labels(service='image').inc()

        # === VIDEO PREDICTION ===
        elif 'video_file' in request.files:
            video_file = request.files['video_file']
            if video_file.filename != '':
                temp_vid = tempfile.NamedTemporaryFile(delete=False)
                temp_vid.write(video_file.read())
                temp_vid.close()

                cap = cv2.VideoCapture(temp_vid.name)
                predicted_sequence = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    label, conf = predict_from_image(frame)
                    if conf > 0.75:
                        predicted_sequence.append(label)

                cap.release()
                os.remove(temp_vid.name)
                sentence = ''.join(predicted_sequence)
                audio_file = speak_from_text(sentence)
                service_usage.labels(service='video').inc()

        # === VOICE TO TEXT ===
        elif request.form.get('mic_input') == 'true':
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                try:
                    audio = recognizer.listen(source, timeout=5)
                    prediction = recognizer.recognize_google(audio)
                    audio_file = speak_from_text(prediction)
                except:
                    prediction = "Couldn't recognize voice"
            service_usage.labels(service='voice').inc()

        # === TEXT TO SPEECH ===
        elif 'tts_input' in request.form:
            text = request.form.get('tts_input')
            if text:
                audio_file = speak_from_text(text)
                prediction = text
            service_usage.labels(service='text').inc()

    return render_template('app.html', prediction=prediction, sentence=sentence, audio_file=audio_file)

@app.route('/live_predict', methods=['POST'])
def live_predict():
    img_file = request.files['image_file']
    label, conf = predict_from_image(img_file)
    service_usage.labels(service='live_predict').inc()
    return jsonify({"label": label, "confidence": conf})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
