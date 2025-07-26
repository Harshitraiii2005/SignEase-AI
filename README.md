# 🤟 SignEase AI – Bridging the Communication Gap

![GitHub last commit](https://img.shields.io/github/last-commit/HarshitRai/SignEaseAI)
![Repo size](https://img.shields.io/github/repo-size/HarshitRai/SignEaseAI)
![Issues](https://img.shields.io/github/issues/HarshitRai/SignEaseAI)
![License](https://img.shields.io/github/license/HarshitRai/SignEaseAI)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue?logo=githubactions)
![Built With Love](https://img.shields.io/badge/built%20with-%E2%9D%A4-red)

> A real-time AI-based assistive platform for the Deaf and Sign Language users, enabling seamless multimodal communication using computer vision and speech technologies.

---

## 🧠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-ML-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-informational?logo=opencv)
![Flask](https://img.shields.io/badge/Flask-API-lightgrey?logo=flask)
![CNN](https://img.shields.io/badge/CNN-Image%20Detection-red)
![RNN](https://img.shields.io/badge/RNN-Voice-green)
![LSTM](https://img.shields.io/badge/LSTM-Sequence-blueviolet)
![Docker](https://img.shields.io/badge/Docker-Container-blue?logo=docker)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Minikube-blue?logo=kubernetes)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-orange?logo=prometheus)
![Grafana](https://img.shields.io/badge/Grafana-Dashboard-yellow?logo=grafana)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue?logo=githubactions)

---

## 🎯 Project Overview

**SignEase AI** is a full-stack MLOps-powered platform that enables:

- 🖼️ Image and 🎞️ video-based sign language translation
- 🎙️ Speech-to-text live assistant
- 🔊 Text-to-speech conversion
- 📸 Live webcam sign detection

Trained using the **American Sign Language (ASL) Dataset**, this system achieves **90%+ model accuracy** and is optimized for real-time responsiveness.

---

## 🚀 Key Features

| Module | Description |
|--------|-------------|
| 🖼️ **Image to Sign Detection** | Upload an image → detect signs using CNN |
| 🎞️ **Video Sign Translation** | Upload video → recognize sign gestures into text/speech |
| 🎙️ **Voice to Text Assistant** | Live voice input → transcribe with RNN |
| 🔊 **Text to Speech** | Type text → generate voice |
| 📸 **Webcam Prediction** | Real-time webcam-based sign prediction |
| 🔬 **Postman-tested APIs** | All routes verified using Postman |
| 📈 **Model Accuracy** | Trained on ASL dataset → **90% accuracy** |
| 🧪 **Real-time Deployment** | Locally hosted via **Minikube + Kubernetes** |
| 🔁 **CI/CD Pipeline** | Automated build & deploy with **GitHub Actions** |
| 📊 **Monitoring** | **Prometheus + Grafana** integrated |

---

## 🧪 Model Details

| Component        | Architecture | Purpose                       |
|------------------|--------------|-------------------------------|
| **Sign Detection** | CNN          | Image & Video gesture extraction |
| **Speech-to-Text** | RNN          | Real-time audio transcription |
| **Sign-to-Text**   | LSTM         | Sequence translation          |

- 📊 **Dataset**: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)  
- ✅ **Accuracy**: ~90%  
- 🧪 **Evaluation**: On validation set, tested on unseen signs  
- 🧪 **API Testing**: Done via Postman (GET/POST)

---

## 🌐 Deployment

| Tool        | Role                             |
|-------------|----------------------------------|
| **Docker**  | Containerization of app & services |
| **Kubernetes (Minikube)** | Local cluster management |
| **Prometheus** | Metrics collection              |
| **Grafana**    | Metrics visualization dashboard |
| **GitHub Actions** | CI/CD automation pipeline   |

### 🔧 Run Locally

```bash
# Clone the repo
git clone https://github.com/HarshitRai/SignEaseAI.git
cd SignEaseAI

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
````

### 🐳 Docker

```bash
docker-compose up --build
```

### ☸️ Minikube + Kubernetes

```bash
minikube start
kubectl apply -f k8s/
```

---

## 📹 Demo

> 🎥 **Watch the full working demo of SignEase AI in action below:**

[![SignEase AI Demo](https://github.com/Harshitraiii2005/SignEase-AI/blob/main/SignEaseAI-GoogleChrome2025-07-2702-05-14-ezgif.com-video-to-gif-converter.gif)

---

📸 Screenshots
🖼️ Explore some core components of SignEase AI through the following snapshots:

🔧 Minikube Kubernetes Service
[![SignEase AI Demo](https://github.com/Harshitraiii2005/SignEase-AI/blob/main/WhatsApp%20Image%202025-07-27%20at%2002.53.26_12fba7c8.jpg)


📮 Postman API Testing
[![SignEase AI Demo](https://github.com/Harshitraiii2005/SignEase-AI/blob/main/http___127.0.0.1_5000_%20-%20Harshit%20Rai's%20Workspace%207_27_2025%202_47_12%20AM.png)

📊 Grafana Monitoring Dashboard
[![SignEase AI Demo](https://github.com/Harshitraiii2005/SignEase-AI/blob/main/first%20one%20-%20Dashboards%20-%20Grafana%20-%20Google%20Chrome%207_27_2025%203_03_52%20AM.png)


## 🧪 Postman API Collection

* `/predict/image` – Upload image and get predicted sign
* `/predict/video` – Upload video and receive transcript
* `/voice-to-text` – Convert voice to readable text
* `/text-to-speech` – Convert input text to audio

> ✅ All APIs tested and validated via Postman

---

## 👨‍💻 Author

Crafted with ❤️ by **Harshit Rai**
📅 **SignEase AI © 2025**
🔗 [LinkedIn](https://www.linkedin.com/in/harshit-rai-5b91142a8/) 

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

