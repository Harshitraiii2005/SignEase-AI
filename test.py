import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

# ========== Step 1: Load Model & Test Data ==========
MODEL_PATH = "Saved_Models/model_07_13_2025_22_12_53.h5"
X_TEST_PATH = "processed_data/x_test.npy"
Y_TEST_PATH = "processed_data/y_test.npy"

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "nothing", "space"
]

SAVE_DIR = "artifacts/metrics"
os.makedirs(SAVE_DIR, exist_ok=True)

print("✅ Loading model and test data...")
model = load_model(MODEL_PATH)
x_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("✅ Model and test data loaded successfully!")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# ========== Step 2: Predict & Evaluate ==========
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Accuracy
accuracy = np.mean(y_true == y_pred_classes) * 100

# Classification Report
report = classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES)
report_dict = classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES, output_dict=True)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# ========== Step 3: Save Results ==========
# Save accuracy
with open(os.path.join(SAVE_DIR, "accuracy.txt"), "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy: {accuracy:.2f}%\n")

# Save classification report (plain text)
with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write("Classification Report:\n")
    f.write(report)

# Save classification report as JSON
with open(os.path.join(SAVE_DIR, "classification_report.json"), "w", encoding="utf-8") as f:
    json.dump(report_dict, f, indent=4)

# Save confusion matrix as JSON
with open(os.path.join(SAVE_DIR, "confusion_matrix.json"), "w", encoding="utf-8") as f:
    json.dump(conf_matrix.tolist(), f, indent=4)

# Save confusion matrix plot
plt.figure(figsize=(16, 14))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
plt.close()

print(f"✅ All metrics and plots saved to: {SAVE_DIR}")
