import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> float:
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Training Decision Tree model")
            dt_clf = DecisionTreeClassifier()
            param_grid = {"criterion": ["gini", "entropy"]}
            
            logging.info("Performing GridSearchCV")
            grid_search = GridSearchCV(dt_clf, param_grid, cv=3, n_jobs=1, verbose=2)
            grid_search.fit(X_train, y_train)
            
            logging.info(f"Best parameters: {grid_search.best_params_}")
            best_model = grid_search.best_estimator_
            
            logging.info("Saving the best model")
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            logging.info("Making predictions with the best model")
            predictions = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            logging.info(f"Model accuracy: {accuracy}")
            return accuracy

        except Exception as e:
            logging.error("An error occurred during model training")
            raise CustomException(e, sys)