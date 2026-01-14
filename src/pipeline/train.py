from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    ingestion_config = DataIngestionConfig()
    ingestion = DataIngestion(ingestion_config)

    train_data, test_data = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_data, test_data
    )

    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_arr, test_arr)


