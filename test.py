import numpy as np

x_train = np.load("Processed_Data/x_train.npy")
y_train = np.load("Processed_Data/y_train.npy")
x_test = np.load("Processed_Data/x_test.npy")
y_test = np.load("Processed_Data/y_test.npy")

print("📊 Loaded shapes:")
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)
