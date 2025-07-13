import os
import logging
import numpy as np

from Constants import x_test_path, x_train_path, y_test_path, y_train_path
from Entity.artifact_entity import DataLoaderArtifact
from Entity.config_entity import DataLoaderConfig
from Utils.Custom_exception import MyException
from Utils.Logger import configure_logger

# Configure logger
configure_logger()

class DataLoader:
    def __init__(self, config: DataLoaderConfig):
        self.config = config

    def load_data(self) -> DataLoaderArtifact:
        try:
            logging.info("📦 Loading processed data from .npy files...")

            x_train_data = np.load(self.config.x_train_path)
            x_test_data = np.load(self.config.x_test_path)
            y_train_data = np.load(self.config.y_train_path)
            y_test_data = np.load(self.config.y_test_path)

            logging.info(f"✅ x_train shape: {x_train_data.shape}")
            logging.info(f"✅ y_train shape: {y_train_data.shape}")
            logging.info(f"✅ x_test shape: {x_test_data.shape}")
            logging.info(f"✅ y_test shape: {y_test_data.shape}")
            logging.info("✅ Data loading completed successfully.")

            return DataLoaderArtifact(
                x_train=x_train_data,
                y_train=y_train_data,
                x_test=x_test_data,
                y_test=y_test_data
            )

        except Exception as e:
            logging.error("❌ Error occurred while loading data.")
            raise MyException(f"Error in DataLoader.load_data: {str(e)}")
