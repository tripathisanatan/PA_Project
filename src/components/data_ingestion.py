import os
import sys
from src.exception import CustomException
from src.logger import logging 
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, str]:
        logging.info("Entered the data ingestion method or component")

        try:
            dataset_path = "notebook/data/heart_attack_youngsters_india.csv"

            # ✅ Check if the dataset exists before proceeding
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset not found at {dataset_path}")

            df = pd.read_csv(dataset_path)
            logging.info("Dataset loaded successfully")

            # ✅ Check if the DataFrame is empty
            if df.empty:
                raise ValueError("Loaded dataset is empty")

            # ✅ Ensure the artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            # ✅ Splitting data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"Train data saved at {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved at {self.ingestion_config.test_data_path}")

            logging.info("Data ingestion completed successfully")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except FileNotFoundError as e:
            logging.error(str(e))
            raise CustomException(e, sys)

        except ValueError as e:
            logging.error(str(e))
            raise CustomException(e, sys)

        except Exception as e:
            logging.error("An unexpected error occurred during data ingestion")
            logging.error(str(e))
            raise CustomException(e, sys)

if __name__ == "__main__":
    # ✅ Data Ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # ✅ Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # ✅ Model Training
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
