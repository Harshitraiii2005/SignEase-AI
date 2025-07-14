import numpy as np
import cv2
import tensorflow as tf

IMG_SIZE = 64

def load_model(path="model/asl_model.h5"):
    return tf.keras.models.load_model(path)

def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(model, img, classes):
    processed = preprocess_image(img)
    prediction = model.predict(processed, verbose=0)[0]
    index = np.argmax(prediction)
    confidence = prediction[index]
    return classes[index], confidence
