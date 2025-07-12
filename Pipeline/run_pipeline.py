import sys
import logging
from Utils.Custom_exception import MyException
from Utils.Logger import configure_logger

from Data_Ingestion.data_getter import GetData
from Data_Ingestion.data_normalizer import DataNormalizer
from Data_Ingestion.data_augmentation import DataAugmentation

from Entity.config_entity import (
    DataGetterConfig,
    DataNormalizerConfig,
    DataAugmentationConfig
)

from Entity.artifact_entity import (
    DataNormalizerArtifact,
    DataAugmentationArtifact
)

# Configure logging once
configure_logger()

class Trainpipeline:
    def __init__(self):
        try:
            logging.info(" Initializing Trainpipeline")

            # Configs
            self.data_getter_config = DataGetterConfig()
            self.data_normalizer_config = DataNormalizerConfig()
            self.data_augmentation_config = DataAugmentationConfig()

            # Components
            self.data_getter = GetData(self.data_getter_config)
            self.data_normalizer = DataNormalizer(self.data_normalizer_config)
            self.data_augmenter = DataAugmentation(self.data_augmentation_config)

        except Exception as e:
            raise MyException(" Failed to initialize Trainpipeline", sys) from e

    def run(self):
        try:
            logging.info(" Starting the training pipeline")

            # Step 1: Get raw data
            data_getter_artifact = self.data_getter.get_data()

            # Step 2: Normalize and split the data
            normalized_artifact = self.data_normalizer.normalize_and_split(
                raw_data=data_getter_artifact.data
            )

            logging.info(f" Data Normalization Completed. Train shape: {normalized_artifact.x_train.shape}, Test shape: {normalized_artifact.x_test.shape}")

            # Step 3: Augment data and save
            augmentation_artifact = self.data_augmenter.augment_data(
                x_train=normalized_artifact.x_train,
                y_train=normalized_artifact.y_train,
                x_test=normalized_artifact.x_test,
                y_test=normalized_artifact.y_test
            )

            logging.info("✅ Data Augmentation Completed and saved to Processed_Data directory.")

            return augmentation_artifact  # or normalized_artifact if needed

        except Exception as e:
            logging.error(f" Error in Trainpipeline: {str(e)}")
            raise MyException("Pipeline execution failed", sys) from e
