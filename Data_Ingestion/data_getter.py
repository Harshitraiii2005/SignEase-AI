import os
import cv2
import numpy as np
import sys
import logging

from Utils.Custom_exception import MyException
from Utils.Logger import configure_logger
from Constants import DATASET_PATH, IMG_SIZE
from Entity.artifact_entity import DataGetterArtifact
from Entity.config_entity import DataGetterConfig

configure_logger()

class GetData:
    def __init__(self, config: DataGetterConfig):
        self.config = config
        self.dataset_path = DATASET_PATH
        self.img_size = IMG_SIZE
        self.classes = os.listdir(self.dataset_path)

    def get_data(self) -> DataGetterArtifact:
        logging.info("Starting data retrieval process")
        try:
            data = []
            for class_name in self.classes:
                class_path = os.path.join(self.dataset_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.png')):
                        img_path = os.path.join(class_path, img_file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.resize(img, (self.img_size, self.img_size))
                            data.append((img, class_name))
            logging.info(f"Retrieved {len(data)} images from dataset")
            return DataGetterArtifact(
                data_shape=(len(data), self.img_size, self.img_size, 3),
                data_class=self.classes,
                data=data
            )
        except Exception as e:
            raise MyException("Data retrieval failed", sys) from e
