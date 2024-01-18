import os
import sys

import numpy as np 
import pandas as pd
import dill  #used to store Python objects to a file, but the primary usage is to send Python objects across the network as a byte stream.
import pickle #pickle module - used for serializing and de-serializing a Python object structure. 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle serialization.

    Parameters:
    - file_path (str): Path to the file where the object will be saved.
    - obj (object): The object to be saved.

    Raises:
    - CustomException: If an error occurs during the saving process.
    """
    try:
        dir_path = os.path.dirname(file_path)

        # Create directories if they don't exist
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the specified file path
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception with details about the error
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate machine learning models using GridSearchCV.

    Parameters:
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data labels.
    - X_test (array-like): Testing data features.
    - y_test (array-like): Testing data labels.
    - models (dict): Dictionary of models to evaluate.
    - param (dict): Dictionary of hyperparameters for each model.

    Returns:
    - report (dict): Dictionary containing model names as keys and R^2 scores as values.

    Raises:
    - CustomException: If an error occurs during the evaluation process.
    """
    try:
        report = {}

        # Loop through the models and their corresponding hyperparameters
        for model_name, model in models.items():
            model_params = param[model_name]

            # Use GridSearchCV to find the best hyperparameters
            gs = GridSearchCV(model, model_params, cv=3)
            gs.fit(X_train, y_train)

            # Set the best hyperparameters to the model
            model.set_params(**gs.best_params_)

            # Fit the model with the best hyperparameters
            model.fit(X_train, y_train)

            # Predictions on training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # R^2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test R^2 score in the report
            report[model_name] = test_model_score

        return report

    except Exception as e:
        # Raise a custom exception with details about the error
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load a Python object from a file using pickle deserialization.

    Parameters:
    - file_path (str): Path to the file from which the object will be loaded.

    Returns:
    - obj (object): The loaded object.

    Raises:
    - CustomException: If an error occurs during the loading process.
    """
    try:
        # Load the object from the specified file path
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        # Raise a custom exception with details about the error
        raise CustomException(e, sys)