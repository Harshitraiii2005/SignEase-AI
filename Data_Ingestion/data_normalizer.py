import sys
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from Utils.Logger import configure_logger
from Utils.Custom_exception import MyException

from Entity.artifact_entity import DataNormalizerArtifact
from Entity.config_entity import DataNormalizerConfig

configure_logger()

class DataNormalizer:
    def __init__(self, config: DataNormalizerConfig):
        self.config = config

    def normalize_and_split(self, raw_data: list) -> DataNormalizerArtifact:
        try:
            logging.info("📦 Starting data normalization and splitting")

            # Step 1: Separate features and labels
            X = [img for img, label in raw_data]
            y = [label for img, label in raw_data]

            X = np.array(X)
            y = np.array(y)

            logging.info(f"✅ Raw data separated: {X.shape}, {y.shape}")

            # Step 2: Normalize images (0–255 → 0–1)
            X = X.astype('float32') / 255.0

            # Step 3: Encode labels
            encoder = LabelBinarizer()
            y_encoded = encoder.fit_transform(y)

            logging.info(f"🔠 Label encoding done. Shape: {y_encoded.shape}")

            # Step 4: Split into train and test
            x_train, x_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=self.config.test_size, random_state=42, stratify=y
            )

            logging.info("✅ Data normalization and splitting completed.")

            return DataNormalizerArtifact(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test
            )

        except Exception as e:
            raise MyException("❌ Failed in normalize_and_split", sys) from e
