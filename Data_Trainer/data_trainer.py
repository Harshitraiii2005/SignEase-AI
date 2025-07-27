import os
import logging
import numpy as np
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization,
    Input, TimeDistributed, LSTM
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from Utils.Custom_exception import MyException
from Utils.Logger import configure_logger
from Entity.artifact_entity import DataLoaderArtifact, DataTrainerArtifact
from Entity.config_entity import ModelTrainerConfig

configure_logger()

class ModelTrainer:
    def __init__(self, data_trainer_artifact: DataTrainerArtifact, model_trainer_config: ModelTrainerConfig):
        self.data_loader_artifact = data_trainer_artifact.data_loader_artifact
        self.config = model_trainer_config
        self.model = self._create_model()

    def _create_model(self):
        """
        Create a CNN + LSTM model for image/video/text sequence classification.
        Assumes input shape: (time_steps, height, width, channels)
        """
        input_shape = self.data_loader_artifact.x_train.shape[1:]  # (timesteps, h, w, c)
        num_classes = self.data_loader_artifact.y_train.shape[1]

        input_layer = Input(shape=input_shape)

        # Apply CNN on each frame using TimeDistributed
        cnn = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(input_layer)
        cnn = TimeDistributed(BatchNormalization())(cnn)
        cnn = TimeDistributed(MaxPooling2D((2, 2)))(cnn)

        cnn = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(cnn)
        cnn = TimeDistributed(BatchNormalization())(cnn)
        cnn = TimeDistributed(MaxPooling2D((2, 2)))(cnn)

        cnn = TimeDistributed(Conv2D(128, (3, 3), activation='relu'))(cnn)
        cnn = TimeDistributed(BatchNormalization())(cnn)
        cnn = TimeDistributed(MaxPooling2D((2, 2)))(cnn)

        cnn = TimeDistributed(Flatten())(cnn)

        # LSTM for sequence modeling after CNN feature extraction
        lstm = LSTM(128, return_sequences=False)(cnn)
        dense = Dense(256, activation='relu')(lstm)
        dropout = Dropout(0.5)(dense)
        output = Dense(num_classes, activation='softmax')(dropout)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train_model(self) -> DataTrainerArtifact:
        """
        Train the CNN+LSTM model and return a DataTrainerArtifact.
        """
        try:
            logging.info("🚀 Starting model training...")

            datagen = ImageDataGenerator(
                rotation_range=15,
                zoom_range=0.2,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            checkpoint = ModelCheckpoint(
                self.config.model_save_path, monitor='val_accuracy',
                save_best_only=True, mode='max', verbose=1
            )

            early_stop = EarlyStopping(
                monitor='val_loss', patience=5, mode='min', verbose=1
            )

            history = self.model.fit(
                datagen.flow(
                    self.data_loader_artifact.x_train,
                    self.data_loader_artifact.y_train,
                    batch_size=self.config.batch_size
                ),
                epochs=self.config.epochs,
                validation_data=(
                    self.data_loader_artifact.x_test,
                    self.data_loader_artifact.y_test
                ),
                callbacks=[checkpoint, early_stop]
            )

            logging.info(f"✅ Model training completed. Model saved at: {self.config.model_save_path}")

            return DataTrainerArtifact(
                data_loader_artifact=self.data_loader_artifact,
                trained_model=self.model,
                model_path=self.config.model_save_path,
                training_history=history.history
            )

        except Exception as e:
            logging.error(f"❌ Error in ModelTrainer: {str(e)}")
            raise MyException("❌ Error during model training", sys) from e
