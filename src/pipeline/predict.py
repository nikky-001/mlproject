import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            logging.error("Error in prediction: %s", e)
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 City: str,
                 PM2_5: float,
                 PM10: float,
                 NO: float,
                 NO2: float,
                 NOx: float,
                 NH3: float,
                 CO: float,
                 SO2: float,
                 O3: float,
                 AQI_Bucket: str,
                 year: int,
                 month: int,
                 day: int
                 ):
        
        self.City = City
        self.PM2_5 = PM2_5
        self.PM10 = PM10
        self.NO = NO
        self.NO2 = NO2
        self.NOx = NOx
        self.NH3 = NH3
        self.CO = CO
        self.SO2 = SO2
        self.O3 = O3
        self.AQI_Bucket = AQI_Bucket    
        self.year = year
        self.month = month
        self.day = day

    def get_data_as_dataframe(self):
        try:
            data = {
                'City': [self.City],
                'PM2.5': [self.PM2_5],
                'PM10': [self.PM10],
                'NO': [self.NO],
                'NO2': [self.NO2],
                'NOx': [self.NOx],
                'NH3': [self.NH3],
                'CO': [self.CO],
                'SO2': [self.SO2],
                'O3': [self.O3],
                'AQI_Bucket': [self.AQI_Bucket],
                'year': [self.year],
                'month': [self.month],
                'day': [self.day]
            }
            df = pd.DataFrame(data)
            logging.info("Dataframe created successfully")
            return df
        except Exception as e:
            logging.error("Error in get_data_as_dataframe: %s", e)
            raise CustomException(e, sys)
       
       