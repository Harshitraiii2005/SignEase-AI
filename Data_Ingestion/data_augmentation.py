import os
import sys
import logging
import numpy as np
from Utils.Custom_exception import MyException
from Utils.Logger import configure_logger
from Entity.artifact_entity import DataAugmentationArtifact
from Entity.config_entity import DataAugmentationConfig
from tensorflow.keras.preprocessing.image import ImageDataGenerator

configure_logger()

class DataAugmentation:
    def __init__(self, config: DataAugmentationConfig):
        self.config = config
        self.datagen = ImageDataGenerator(
            rotation_range=config.augmentation_params['rotation_range'],
            width_shift_range=config.augmentation_params['width_shift_range'],
            height_shift_range=config.augmentation_params['height_shift_range'],
            zoom_range=config.augmentation_params['zoom_range'],
            horizontal_flip=config.augmentation_params['horizontal_flip']
        )

    def _ensure_dir(self, file_path):
        """Ensure directory exists before saving a file."""
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

    def _augment_and_combine(self, x, y, label):
        self.datagen.fit(x)
        aug_iterator = self.datagen.flow(x, y, batch_size=32)
        x_aug, y_aug = [], []

        for _ in range(len(x) // 32):
            x_batch, y_batch = next(aug_iterator)
            x_aug.append(x_batch)
            y_aug.append(y_batch)

        x_aug = np.vstack(x_aug)
        y_aug = np.vstack(y_aug)

        combined_x = np.concatenate([x, x_aug])
        combined_y = np.concatenate([y, y_aug])

        logging.info(f"✅ {label} augmentation done. Original: {x.shape}, Final: {combined_x.shape}")
        return combined_x, combined_y

    def augment_data(self, x_train: np.ndarray, y_train: np.ndarray,
                     x_test: np.ndarray, y_test: np.ndarray) -> DataAugmentationArtifact:
        try:
            logging.info("📦 Starting data augmentation for train and test sets")

            # Augment training set
            x_train_aug, y_train_aug = self._augment_and_combine(x_train, y_train, "Train")

            # Augment test set
            x_test_aug, y_test_aug = self._augment_and_combine(x_test, y_test, "Test")

            # Ensure directories exist before saving
            self._ensure_dir(self.config.x_train_path)
            self._ensure_dir(self.config.y_train_path)
            self._ensure_dir(self.config.x_test_path)
            self._ensure_dir(self.config.y_test_path)

            # Save all
            np.save(self.config.x_train_path, x_train_aug)
            np.save(self.config.y_train_path, y_train_aug)
            np.save(self.config.x_test_path, x_test_aug)
            np.save(self.config.y_test_path, y_test_aug)

            logging.info("✅ All augmented data saved successfully")

            return DataAugmentationArtifact(
                x_train_augmented=x_train_aug,
                y_train_augmented=y_train_aug,
                x_test_augmented=x_test_aug,
                y_test_augmented=y_test_aug
            )

        except Exception as e:
            raise MyException("❌ Failed in augment_data (train/test)", sys) from e
