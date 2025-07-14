// static/predictions.js
let video = document.getElementById('webcam');
let canvas = document.createElement('canvas');
let predictionLabel = document.getElementById('live-label');

function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            video.play();
            setInterval(captureFrame, 1000); // every 1s
        });
}

function captureFrame() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('image_file', blob, 'frame.jpg');

        fetch('/live_predict', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            predictionLabel.innerText = `Prediction: ${data.label} (${data.confidence})`;
        });
    }, 'image/jpeg');
}
