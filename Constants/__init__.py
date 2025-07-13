import os

# Dataset Paths
DATASET_PATH = os.path.join("Dataset", "asl_alphabet_train")
TEST_DATASET_PATH = os.path.join("Dataset", "asl_alphabet_test")
IMG_SIZE = 64
IMG_SIZE_TEST = 64

# Class names
Classes = os.listdir(DATASET_PATH)

# Processed Data Save Paths
PROCESSED_DATA_DIR = os.path.join("Processed_Data")
x_train_path = os.path.join(PROCESSED_DATA_DIR, "x_train.npy")
x_test_path = os.path.join(PROCESSED_DATA_DIR, "x_test.npy")
y_train_path = os.path.join(PROCESSED_DATA_DIR, "y_train.npy")
y_test_path = os.path.join(PROCESSED_DATA_DIR, "y_test.npy")


