import logging
import sys
import numpy as np

# Configure logging format
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Import modules
from Data_Ingestion.data_getter import GetData
from Data_Ingestion.data_normalizer import DataNormalizer
from Data_Ingestion.data_augmentation import DataAugmentation
from Constants import *
from Entity.config_entity import (
    DataGetterConfig,
    DataNormalizerConfig,
    DataAugmentationConfig
)
def main():
    try:
        logging.info("🚦 Pipeline started")

        # 1. Data Ingestion
        getter = GetData()  # Pass config if needed
        data_artifact = getter.get_data()
        logging.info(f"✅ Data ingestion complete: {data_artifact.data_shape[0]} samples")

        # 2. Data Normalization & Splitting
        normalizer = DataNormalizer(test_size=0.2)
        norm_artifact = normalizer.normalize_and_split(data_artifact.data)
        logging.info(f"✅ Data normalized & split: {norm_artifact.x_train.shape[0]} train, {norm_artifact.x_test.shape[0]} test")

        # 3. Data Augmentation (training set only)
        augment_config = DataAugmentationConfig()
        augmenter = DataAugmentation(augment_config.augmentation_params)
        aug_artifact = augmenter.augment_data(
            norm_artifact.x_train,
            norm_artifact.y_train,
            norm_artifact.x_test,
            norm_artifact.y_test
        )
        logging.info("✅ Data augmentation complete (train only)")

        # 4. Save Artifacts
        np.save(X_TRAIN_PATH, aug_artifact.x_train_augmented)
        np.save(Y_TRAIN_PATH, aug_artifact.y_train_augmented)
        np.save(X_TEST_PATH, aug_artifact.x_test_augmented)
        np.save(Y_TEST_PATH, aug_artifact.y_test_augmented)
        logging.info(f"💾 Processed arrays saved in directory: {X_TRAIN_PATH.rsplit('/', 1)[0]}")

        logging.info("🎉 Pipeline finished successfully! Ready for model training.")

    except Exception as e:
        logging.error(f"❌ Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
