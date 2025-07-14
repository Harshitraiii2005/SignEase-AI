import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io

# Dummy labels just to simulate result
DUMMY_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['space', 'nothing']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def predict_from_image(image_file):
    if isinstance(image_file, (str, bytes, bytearray)):
        img = Image.open(io.BytesIO(image_file.read()))
    elif isinstance(image_file, np.ndarray):
        img = Image.fromarray(image_file)
    else:
        img = Image.open(image_file)

    img = img.convert('RGB')
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        # Simulate prediction (always return 'A' for now)
        return "A", 0.90  # Confidence is hardcoded
    else:
        return "No Hand Detected", 0.0
