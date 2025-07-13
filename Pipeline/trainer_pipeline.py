import sys
import logging
from Utils.Logger import configure_logger
from Utils.Custom_exception import MyException

from Data_Trainer.data_loader import DataLoader
from Data_Trainer.data_trainer import ModelTrainer

from Entity.config_entity import DataLoaderConfig, ModelTrainerConfig
from Entity.artifact_entity import DataTrainerArtifact

# Setup logger
configure_logger()

class ModelTrainerPipeline:
    def __init__(self):
        try:
            logging.info("🧠 Initializing ModelTrainerPipeline")

            # Step 1: Initialize DataLoader
            self.data_loader_config = DataLoaderConfig()
            self.data_loader = DataLoader(self.data_loader_config)

            # Step 2: Initialize ModelTrainerConfig
            self.model_trainer_config = ModelTrainerConfig()

        except Exception as e:
            raise MyException("❌ Failed to initialize ModelTrainerPipeline", sys) from e

    def run(self) -> DataTrainerArtifact:
        try:
            logging.info("🚀 Starting the Model Training pipeline")

            # Step 1: Load processed data
            data_loader_artifact = self.data_loader.load_data()

            logging.info("✅ Data Loaded Successfully!")
            logging.info(f"📊 Shapes:")
            logging.info(f"🔹 x_train: {data_loader_artifact.x_train.shape}")
            logging.info(f"🔹 y_train: {data_loader_artifact.y_train.shape}")
            logging.info(f"🔹 x_test : {data_loader_artifact.x_test.shape}")
            logging.info(f"🔹 y_test : {data_loader_artifact.y_test.shape}")

            # Step 2: Prepare training artifact
            dummy_artifact = DataTrainerArtifact(
                data_loader_artifact=data_loader_artifact,
                trained_model=None,       # will be set after training
                model_path="",            # will be set after training
                training_history={}       # will be set after training
            )

            # Step 3: Train model
            model_trainer = ModelTrainer(dummy_artifact, self.model_trainer_config)
            final_trainer_artifact = model_trainer.train_model()

            logging.info("✅ Model Trainer Pipeline executed successfully!")
            logging.info(f"📁 Model saved at: {final_trainer_artifact.model_path}")

            return final_trainer_artifact

        except Exception as e:
            raise MyException("❌ Failed during ModelTrainerPipeline execution", sys) from e
