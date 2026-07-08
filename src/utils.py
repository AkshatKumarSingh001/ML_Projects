'''
Utils.py is a utility module that contains helper functions and classes used throughout the project. It provides common functionalities that can be reused in different parts of the codebase.

'''

import pickle
import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.exception import CustomException


def save_object(file_path, obj):
    '''
    This function saves a Python object to a specified file path using the pickle module.
    
    Args:
        file_path (str): The path where the object will be saved.
        obj (object): The Python object to be saved.
    
    Raises:
        CustomException: If there is an error while saving the object.
    '''
    try:
        dir_path = os.path.dirname(file_path) # Get the directory path from the file path.
        os.makedirs(dir_path, exist_ok=True) # Create the directory if it doesn't exist.

        with open(file_path, 'wb') as file_obj: # Open the file in write-binary mode.
            dill.dump(obj, file_obj) # Save the object to the file using pickle.

    except Exception as e:
        raise CustomException(e, sys) # Raise a custom exception if any error occurs while saving the object.
    
def evaluate_models(_X_train,y_train,X_test,y_test,models):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            _X_train,y_train,test_size=0.2,random_state=42
        )

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,y_train) # Fit the model on the training data.

            y_train_pred = model.predict(X_train) # Predict on the training data.
            y_test_pred = model.predict(X_test) # Predict on the test data.

            train_model_score = r2_score(y_train,y_train_pred) # Calculate R2 score for training data.
            test_model_score = r2_score(y_test,y_test_pred) # Calculate R2 score for test data.

            report[list(models.keys())[i]] = test_model_score # Store the test score in the report dictionary.

        return report
        
    except Exception as e:
        raise CustomException(e,sys) # Raise a custom exception if any error occurs during model evaluation.
        
