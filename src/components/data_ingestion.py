import os
import sys
from src.exception import CustomException
from src.logger import logger
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

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
            logger.info("Starting data ingestion...")
            
            df = pd.read_csv(self.config.raw_data_path)
            logger.info("Read the dataset successfully.")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            df.to_csv(self.config.raw_data_path, index=False)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logger.info("Split the dataset into train and test sets.")

            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)
            logger.info("Data ingestion completed successfully.")

            return(
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            logger.error(f"Error occurred during data ingestion: {e}")
            raise CustomException(e,sys)




     