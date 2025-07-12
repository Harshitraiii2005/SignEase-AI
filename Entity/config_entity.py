import os
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
from Constants import DATASET_PATH, IMG_SIZE, PROCESSED_DATA_DIR
from Constants import PROCESSED_DATA_DIR
# Optional: For timestamping artifacts
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class DataGetterConfig:
    data_getter_dir: str = os.path.join('Data_Ingestion')
    data_getter_file_name: str = 'data_getter.py'
    data_getter_object: str = 'GetData'
    data_getter_artifact_dir: str = os.path.join('Artifact', 'data_getter')

@dataclass
class DataNormalizerConfig:
    data_normalizer_dir: str = os.path.join('Data_Transformation')
    data_normalizer_file_name: str = 'data_normalizer.py'
    data_normalizer_object: str = 'DataNormalizer'
    data_normalizer_artifact_dir: str = os.path.join('Artifact', 'data_normalizer')
    
    test_size: float = 0.2  # 80% train / 20% test

@dataclass
class DataAugmentationConfig:
    data_augmentation_dir: str = os.path.join('Data_Ingestion')
    data_augmentation_file_name: str = 'data_augmentation.py'
    data_augmentation_object: str = 'DataAugmentation'
    data_augmentation_artifact_dir: str = os.path.join('Artifact', 'data_augmentation')

    # ✅ FIX: Use field(default_factory=...) for mutable default (dict)
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
