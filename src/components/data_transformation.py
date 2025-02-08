import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "Age", "Screen Time (hrs/day)", "Sleep Duration (hrs/day)",
                "Cholesterol Levels (mg/dL)", "BMI (kg/m²)", "Resting Heart Rate (bpm)",
                "Maximum Heart Rate Achieved", "Blood Oxygen Levels (SpO2%)", 
                "Triglyceride Levels (mg/dL)"
            ]
            
            categorical_columns = [
                "Gender", "Region", "Urban/Rural", "SES", "Smoking Status",
                "Alcohol Consumption", "Diet Type", "Physical Activity Level",
                "Family History of Heart Disease", "Diabetes", "Hypertension",
                "Stress Level", "Blood Pressure (systolic/diastolic mmHg)",
                "ECG Results", "Chest Pain Type", "Exercise Induced Angina"
            ]
            
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ("scaler", StandardScaler(with_mean=False))
            ])
            
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ], remainder='drop')  
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")

            target_column_name = "Heart Attack Likelihood"

            if target_column_name not in train_df.columns or target_column_name not in test_df.columns:
                raise KeyError(f"Target column '{target_column_name}' not found in dataset")

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[[target_column_name]].values.reshape(-1, 1)

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[[target_column_name]].values.reshape(-1, 1)

            logging.info(f"Train input shape before transformation: {input_feature_train_df.shape}")
            logging.info(f"Test input shape before transformation: {input_feature_test_df.shape}")

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Transformed Train input shape: {input_feature_train_arr.shape}")
            logging.info(f"Transformed Test input shape: {input_feature_test_arr.shape}")

            # Ensure the target array has 2 dimensions before concatenation
            if target_feature_train_df.ndim == 1:
                target_feature_train_df = target_feature_train_df.reshape(-1, 1)
            if target_feature_test_df.ndim == 1:
                target_feature_test_df = target_feature_test_df.reshape(-1, 1)

            train_arr = np.hstack((input_feature_train_arr, target_feature_train_df))
            test_arr = np.hstack((input_feature_test_arr, target_feature_test_df))

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)
