import os
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    try:
        # Get form data with validation
        form_data = {
            'age': int(request.form.get('age', 0)),
            'gender': request.form.get('gender', ''),
            'region': request.form.get('region', ''),
            'urban_rural': request.form.get('urban_rural', ''),
            'ses': request.form.get('ses', ''),
            'smoking_status': request.form.get('smoking_status', ''),
            'alcohol_consumption': request.form.get('alcohol_consumption', ''),
            'diet_type': request.form.get('diet_type', ''),
            'physical_activity_level': request.form.get('physical_activity_level', ''),
            'screen_time': int(request.form.get('screen_time', 0)),
            'sleep_duration': int(request.form.get('sleep_duration', 0)),
            'family_history': request.form.get('family_history', ''),
            'diabetes': request.form.get('diabetes', ''),
            'hypertension': request.form.get('hypertension', ''),
            'cholesterol': int(request.form.get('cholesterol', 0)),
            'bmi': float(request.form.get('bmi', 0.0)),
            'stress_level': request.form.get('stress_level', ''),
            'blood_pressure': request.form.get('blood_pressure', ''),
            'resting_heart_rate': int(request.form.get('resting_heart_rate', 0)),
            'ecg_results': request.form.get('ecg_results', ''),
            'chest_pain_type': request.form.get('chest_pain_type', ''),
            'max_heart_rate': int(request.form.get('max_heart_rate', 0)),
            'exercise_induced_angina': request.form.get('exercise_induced_angina', ''),
            'blood_oxygen': float(request.form.get('blood_oxygen', 0.0)),
            'triglycerides': int(request.form.get('triglycerides', 0))
        }

        # Validate required fields
        required_fields = ['age', 'gender', 'bmi']  # Add all required fields
        for field in required_fields:
            if not form_data[field]:
                logger.error(f"Missing required field: {field}")
                return render_template('home.html', error=f"Please fill in the {field} field")

        logger.info("Creating CustomData instance with form data")
        data = CustomData(**form_data)

        logger.info("Converting data to DataFrame")
        pred_df = data.get_data_as_data_frame()
        logger.info(f"Input Data:\n{pred_df}")

        logger.info("Initializing prediction pipeline")
        predict_pipeline = PredictPipeline()
        
        logger.info("Making prediction")
        results = predict_pipeline.predict(pred_df)
        
        logger.info(f"Prediction result: {results[0]}")
        return render_template('home.html', results=results[0])

    except ValueError as e:
        logger.error(f"Invalid input value: {str(e)}")
        return render_template('home.html', error="Please check your input values and try again")
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('home.html', error="An error occurred during prediction")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )