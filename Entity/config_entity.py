import os
from dataclasses import dataclass, field
from datetime import datetime

# Constants
from Constants import DATASET_PATH, IMG_SIZE, PROCESSED_DATA_DIR

# Timestamp for versioning
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


# ---------------- Data Getter Config ----------------
@dataclass
class DataGetterConfig:
    data_getter_dir: str = os.path.join('Data_Ingestion')
    data_getter_file_name: str = 'data_getter.py'
    data_getter_object: str = 'GetData'
    data_getter_artifact_dir: str = os.path.join('Artifact', 'data_getter')


# ---------------- Data Normalizer Config ----------------
@dataclass
class DataNormalizerConfig:
    data_normalizer_dir: str = os.path.join('Data_Transformation')
    data_normalizer_file_name: str = 'data_normalizer.py'
    data_normalizer_object: str = 'DataNormalizer'
    data_normalizer_artifact_dir: str = os.path.join('Artifact', 'data_normalizer')

    test_size: float = 0.2  # Train/Test split ratio


# ---------------- Data Augmentation Config ----------------
@dataclass
class DataAugmentationConfig:
    data_augmentation_dir: str = os.path.join('Data_Ingestion')
    data_augmentation_file_name: str = 'data_augmentation.py'
    data_augmentation_object: str = 'DataAugmentation'
    data_augmentation_artifact_dir: str = os.path.join('Artifact', 'data_augmentation')

    augmentation_params: dict = field(default_factory=lambda: {
        'rotation_range': 20,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'zoom_range': 0.2,
        'horizontal_flip': True
    })

    x_train_path: str = os.path.join(PROCESSED_DATA_DIR, "x_train.npy")
    x_test_path: str = os.path.join(PROCESSED_DATA_DIR, "x_test.npy")
    y_train_path: str = os.path.join(PROCESSED_DATA_DIR, "y_train.npy")
    y_test_path: str = os.path.join(PROCESSED_DATA_DIR, "y_test.npy")


# ---------------- Data Loader Config ----------------
@dataclass
class DataLoaderConfig:
    data_loader_dir: str = os.path.join('Data_Trainer')
    data_loader_file_name: str = 'data_loader.py'
    data_loader_object: str = 'DataLoader'
    data_loader_artifact_dir: str = os.path.join('Artifact', 'data_loader')

    x_train_path: str = os.path.join(PROCESSED_DATA_DIR, "x_train.npy")
    x_test_path: str = os.path.join(PROCESSED_DATA_DIR, "x_test.npy")
    y_train_path: str = os.path.join(PROCESSED_DATA_DIR, "y_train.npy")
    y_test_path: str = os.path.join(PROCESSED_DATA_DIR, "y_test.npy")


# ---------------- Model Trainer Config ----------------
@dataclass
class ModelTrainerConfig:
    model_dir: str = os.path.join('Data_Trainer')              # Directory containing trainer code
    model_dir_name: str = 'data_trainer.py'                    # Training script filename
    model_object: str = 'ModelTrainer'                          # Class name inside model_trainer.py
    model_artifact_dir: str = os.path.join('Artifact', 'model_trainer')  # Output artifact directory

    model_save_path: str = os.path.join('Saved_Models', f'model_{TIMESTAMP}.h5')  # Path to save trained model

    # Training hyperparameters
    epochs: int = 20
    batch_size: int = 32
    validation_split: float = 0.2

    # Optional (you can use these in your model trainer)
    input_shape: tuple = (64, 64, 3)
    num_classes: int = 28
