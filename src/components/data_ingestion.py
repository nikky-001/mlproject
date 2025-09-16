import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion...")
            
            df = pd.read_csv(os.path.join('src', 'notebook', 'data', 'Air_quality_data.csv'))
            logging.info("Read the dataset successfully.")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            df.to_csv(self.config.raw_data_path, index=False)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Split the dataset into train and test sets.")

            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)
            logging.info("Data ingestion completed successfully.")

            return(
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {e}")
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj = DataIngestion(config=DataIngestionConfig())
    train_data,test_data= obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,preprocessor_path= data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

     