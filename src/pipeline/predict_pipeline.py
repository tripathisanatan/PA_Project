import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            logger.info("Loading model and preprocessor")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logger.info("Model and preprocessor loaded successfully")
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            # Convert prediction to human-readable format
            return ["High Risk" if pred == 1 else "Low Risk" for pred in preds]
        
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        age: int,
        gender: str,
        region: str,
        urban_rural: str,
        ses: str,
        smoking_status: str,
        alcohol_consumption: str,
        diet_type: str,
        physical_activity_level: str,
        screen_time: int,
        sleep_duration: int,
        family_history: str,
        diabetes: str,
        hypertension: str,
        cholesterol: int,
        bmi: float,
        stress_level: str,
        blood_pressure: str,
        resting_heart_rate: int,
        ecg_results: str,
        chest_pain_type: str,
        max_heart_rate: int,
        exercise_induced_angina: str,
        blood_oxygen: float,
        triglycerides: int
    ):
        # Input validation
        if not (18 <= age <= 35):
            raise ValueError("Age must be between 18 and 35")
        if bmi <= 0:
            raise ValueError("BMI must be positive")
        if blood_oxygen < 0 or blood_oxygen > 100:
            raise ValueError("Blood oxygen must be between 0 and 100")
        
        self.age = age
        self.gender = gender
        self.region = region
        self.urban_rural = urban_rural
        self.ses = ses
        self.smoking_status = smoking_status
        self.alcohol_consumption = alcohol_consumption
        self.diet_type = diet_type
        self.physical_activity_level = physical_activity_level
        self.screen_time = screen_time
        self.sleep_duration = sleep_duration
        self.family_history = family_history
        self.diabetes = diabetes
        self.hypertension = hypertension
        self.cholesterol = cholesterol
        self.bmi = bmi
        self.stress_level = stress_level
        self.blood_pressure = blood_pressure
        self.resting_heart_rate = resting_heart_rate
        self.ecg_results = ecg_results
        self.chest_pain_type = chest_pain_type
        self.max_heart_rate = max_heart_rate
        self.exercise_induced_angina = exercise_induced_angina
        self.blood_oxygen = blood_oxygen
        self.triglycerides = triglycerides

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.age],
                "Gender": [self.gender],
                "Region": [self.region],
                "Urban/Rural": [self.urban_rural],
                "SES": [self.ses],
                "Smoking Status": [self.smoking_status],
                "Alcohol Consumption": [self.alcohol_consumption],
                "Diet Type": [self.diet_type],
                "Physical Activity Level": [self.physical_activity_level],
                "Screen Time (hrs/day)": [self.screen_time],
                "Sleep Duration (hrs/day)": [self.sleep_duration],
                "Family History of Heart Disease": [self.family_history],
                "Diabetes": [self.diabetes],
                "Hypertension": [self.hypertension],
                "Cholesterol Levels (mg/dL)": [self.cholesterol],
                "BMI (kg/m²)": [self.bmi],
                "Stress Level": [self.stress_level],
                "Blood Pressure (systolic/diastolic mmHg)": [self.blood_pressure],
                "Resting Heart Rate (bpm)": [self.resting_heart_rate],
                "ECG Results": [self.ecg_results],
                "Chest Pain Type": [self.chest_pain_type],
                "Maximum Heart Rate Achieved": [self.max_heart_rate],
                "Exercise Induced Angina": [self.exercise_induced_angina],
                "Blood Oxygen Levels (SpO2%)": [self.blood_oxygen],
                "Triglyceride Levels (mg/dL)": [self.triglycerides]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logger.info("DataFrame Created Successfully")
            return df

        except Exception as e:
            logger.error(f"Error in creating DataFrame: {str(e)}")
            raise CustomException(e, sys)