
# 🤟 SignEase AI – Bridging the Communication Gap

![GitHub last commit](https://img.shields.io/github/last-commit/HarshitRai/SignEaseAI)
![Repo size](https://img.shields.io/github/repo-size/HarshitRai/SignEaseAI)
![Issues](https://img.shields.io/github/issues/HarshitRai/SignEaseAI)
![License](https://img.shields.io/github/license/HarshitRai/SignEaseAI)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue?logo=githubactions)
![Built With Love](https://img.shields.io/badge/built%20with-%E2%9D%A4-red)

> A real-time AI-based assistive platform empowering Deaf and Sign Language users through multimodal communication—leveraging computer vision and speech technologies.

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

---

## 🎯 Project Overview

**SignEase AI** is an MLOps-integrated platform designed to:

* 🖼️ Translate sign language from images and videos
* 🎙️ Convert voice to text in real time
* 🔊 Convert text to speech
* 📸 Detect signs via live webcam

> Trained on the **American Sign Language (ASL) Dataset**, the system achieves over **90% accuracy** and is optimized for real-time performance.

---

## 🚀 Key Features

| Module                   | Description                               |
| ------------------------ | ----------------------------------------- |
| 🖼️ Image Sign Detection | Detect signs from static images using CNN |
| 🎞️ Video Translation    | Recognize continuous signs from videos    |
| 🎙️ Voice-to-Text        | Transcribe spoken input using RNN         |
| 🔊 Text-to-Speech        | Convert typed text to audio output        |
| 📸 Webcam Prediction     | Live webcam-based sign prediction         |
| 🧪 Postman API Tests     | All routes tested with Postman            |
| 📈 Model Accuracy        | Achieves \~90% accuracy on ASL dataset    |
| 🔁 CI/CD Integration     | Automated via GitHub Actions              |
| 📊 Monitoring Dashboard  | Live metrics via Prometheus + Grafana     |

---

## 🧠 Model Architecture

| Component           | Architecture | Purpose                            |
| ------------------- | ------------ | ---------------------------------- |
| Sign Detection      | CNN          | Image/Video gesture classification |
| Speech-to-Text      | RNN          | Real-time audio transcription      |
| Sign Sequence Model | LSTM         | Sign-to-text temporal translation  |

* **Dataset**: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
* **Accuracy**: \~90%
* **Validation**: Evaluated on unseen gestures
* **API Testing**: Verified using Postman (GET/POST routes)

---

## 🌐 Deployment Architecture

| Tool               | Role                               |
| ------------------ | ---------------------------------- |
| **Docker**         | Containerizes all services         |
| **Minikube**       | Local Kubernetes cluster           |
| **Prometheus**     | Metrics collection                 |
| **Grafana**        | Monitoring dashboard visualization |
| **GitHub Actions** | CI/CD automation                   |

---

## 🛠️ Run Locally

```bash
# Clone the repository
git clone https://github.com/HarshitRai/SignEaseAI.git
cd SignEaseAI

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
```

---

## 🐳 Run with Docker

```bash
docker-compose up --build
```

---

## ☸️ Run with Minikube

```bash
minikube start
kubectl apply -f k8s/
```

---

## 📹 Live Demo

🎥 **Watch the working demo of SignEase AI:**

![SignEase Demo](https://github.com/Harshitraiii2005/SignEase-AI/blob/main/SignEaseAI-GoogleChrome2025-07-2702-05-14-ezgif.com-video-to-gif-converter.gif)

---

## 📸 Screenshots

### 🔧 Kubernetes Dashboard (Minikube)

![K8s Dashboard](https://github.com/Harshitraiii2005/SignEase-AI/blob/main/WhatsApp%20Image%202025-07-27%20at%2002.53.26_12fba7c8.jpg)

### 📮 Postman API Testing

![Postman API](https://github.com/Harshitraiii2005/SignEase-AI/blob/main/http___127.0.0.1_5000_%20-%20Harshit%20Rai's%20Workspace%207_27_2025%202_47_12%20AM.png)

### 📊 Grafana Monitoring

![Grafana Dashboard](https://github.com/Harshitraiii2005/SignEase-AI/blob/main/first%20one%20-%20Dashboards%20-%20Grafana%20-%20Google%20Chrome%207_27_2025%203_03_52%20AM.png)

---

## 🔌 API Endpoints

| Endpoint          | Description                         |
| ----------------- | ----------------------------------- |
| `/predict/image`  | Upload image and get predicted sign |
| `/predict/video`  | Upload video for sign recognition   |
| `/voice-to-text`  | Convert voice to text               |
| `/text-to-speech` | Convert text input to speech        |

> ✅ All APIs tested via Postman and ready for integration

---

## 👨‍💻 Author

Made with ❤️ by **Harshit Rai**
📅 **© 2025 – SignEase AI**
🔗 [LinkedIn](https://www.linkedin.com/in/harshit-rai-5b91142a8/)
🌐 [GitHub](https://github.com/HarshitRai)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
