from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class DataGetterArtifact:
    data_shape: Tuple[int, int, int, int]  # Example: (84000, 64, 64, 3)
    data_class: List[str]
    data: List[Tuple[np.ndarray, str]]

@dataclass
class DataNormalizerArtifact:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

@dataclass
class DataAugmentationArtifact:
    x_train_augmented: np.ndarray
    y_train_augmented: np.ndarray
    x_test_augmented: np.ndarray
    y_test_augmented: np.ndarray