from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class DataGetterArtifact:
    data_shape: Tuple[int, int, int, int]
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



@dataclass
class DataLoaderArtifact:
    """
    Final processed data artifact loaded from saved .npy files.
    """
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


@dataclass
class DataTrainerArtifact:
    """
    Artifact generated after model training.
    """
    data_loader_artifact: DataLoaderArtifact
    trained_model: Any                      # Trained Keras model or similar
    model_path: str                         # Path to saved model
    training_history: Dict[str, List[float]]  # History (loss, acc, val_loss, val_acc, etc.)
