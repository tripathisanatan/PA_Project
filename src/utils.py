import os
import sys
import pickle
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    """Save a Python object to a file using pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """Evaluate multiple models using GridSearchCV and return performance metrics."""
    try:
        report = {}
        for model_name, model in models.items():
            params = param.get(model_name, {})
            if params:
                gs = GridSearchCV(model, params, cv=3, scoring='accuracy')
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
            
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else None
            
            report[model_name] = {
                "Train Accuracy": train_accuracy,
                "Test Accuracy": test_accuracy,
                "Test F1-Score": test_f1,
                "Test ROC-AUC": test_roc_auc
            }
        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Load a Python object from a file using pickle."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
