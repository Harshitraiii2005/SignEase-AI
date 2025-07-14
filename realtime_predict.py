import cv2
import numpy as np
import tensorflow as tf
import time
import pyttsx3

# Initialize Text-to-Speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech

# Load trained model
model = tf.keras.models.load_model("Saved_Models/model_07_13_2025_22_12_53.h5")

# ASL classes (must match training labels)
classes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z', 'nothing', 'space'
]

IMG_SIZE = 64
sentence = ""
last_letter = ""
last_update_time = 0
cooldown_time = 1.2  # seconds
CONFIDENCE_THRESHOLD = 0.8

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    x1, y1, x2, y2 = 100, 100, 324, 324
    roi = frame[y1:y2, x1:x2]

    # Preprocess
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)[0]
    predicted_label = classes[np.argmax(prediction)]
    confidence = np.max(prediction)

    current_time = time.time()
    if predicted_label != last_letter or (current_time - last_update_time) > cooldown_time:
        if confidence > CONFIDENCE_THRESHOLD:
            if predicted_label == "space":
                sentence += " "
            elif predicted_label != "nothing":
                sentence += predicted_label
            last_letter = predicted_label
            last_update_time = current_time

    # Draw
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{predicted_label} ({confidence*100:.1f}%)", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, f"Sentence: {sentence}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, "Press 'C' to clear | 'S' to speak | 'Q' to quit", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Sign Language to Sentence with Voice", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""
    elif key == ord('s') and sentence.strip() != "":
        tts_engine.say(sentence)
        tts_engine.runAndWait()

cap.release()
cv2.destroyAllWindows()
