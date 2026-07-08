''' This is the data transformation component which will take the train and test data as input from the data ingestion component,
and will perform the necessary transformation like encoding categorical variables, handling missing values, Data Cleaning etc.
It will return the transformed train and test data for model training and evaluation. 
'''

''' @dataclass exists to eliminate repetitive boilerplate for classes whose only job is to hold data, not behave.
Writing __init__, __repr__, and __eq__ by hand for a simple container class is repetitive and error-prone — you're manually assigning self.x = x for every field, every time.
@dataclass looks at your type-hinted class attributes and generates all of that for you.
'''

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Same pattern as data_ingestion.py, so this file's paths are absolute no matter what directory the script is run from.

@dataclass # dataclass is a decorator that automatically generates special methods like __init__() and __repr__() for the class, based on the class attributes.
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(PROJECT_ROOT, 'artifacts', 'preprocessor.pkl') # Path to save the preprocessor object after transformation. Now absolute, so it always lands in the project-root artifacts/ folder.

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() # Create an instance of the DataTransformationConfig class to access the preprocessor object file path.

    def get_data_transformer_object(self): # Method of the class (not nested inside __init__), so it can be called on any instance via self.
        '''
        This function is responsible for performing data transformation on the train and test data.
        It handles missing values, encodes categorical variables, and scales numerical features.
        
        '''
        try:
            numerical_columns = ['writing_score', 'reading_score'] # List of numerical columns in the dataset.
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ] # List of categorical columns in the dataset.

            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')), # Impute missing values in numerical columns with the median value.
                    ('scaler', StandardScaler()) # Scale numerical columns using StandardScaler (mean-centered, since numerical output is dense).
                ]
            ) # Pipeline that chains imputation and scaling for numerical columns.

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute missing values in categorical columns with the most frequent value.
                    ('one_hot_encoder', OneHotEncoder()), # Encode categorical columns using One-Hot Encoding.
                    ('scaler', StandardScaler(with_mean=False)) # Scale categorical columns without centering, since one-hot output is sparse and mean-centering would break it.
                ]
            ) # Pipeline that chains imputation, encoding, and scaling for categorical columns.

            logging.info(f"Categorical columns standard scaling completed: {categorical_columns}") # Log the categorical columns.
            logging.info(f"Numerical columns standard scaling completed: {numerical_columns}") # Log the numerical columns.

            preprocessor = ColumnTransformer( # Combining both numerical and categorical pipelines into a single preprocessor object.
                [
                    ('num_pipeline', numerical_pipeline, numerical_columns), # Apply the numerical pipeline to the numerical columns.
                    ('cat_pipeline', categorical_pipeline, categorical_columns) # Apply the categorical pipeline to the categorical columns.
                ]
            ) # ColumnTransformer applies each pipeline only to its designated columns, then concatenates the results.

            return preprocessor # Return the preprocessor object.

        except Exception as e:
            raise CustomException(e, sys) # Raise a custom exception if any error occurs during the transformation process.

    def initiate_data_transformation(self, train_path, test_path): # Method of the class (not nested inside __init__), so it's callable as data_transformation.initiate_data_transformation(...).
        try:
            train_df = pd.read_csv(train_path) # Read the training data from the specified path.
            test_df = pd.read_csv(test_path) # Read the testing data from the specified path.

            logging.info("Read train and test data completed") # Log that the data has been read successfully.

            logging.info("Obtaining preprocessing object") # Log that the preprocessing object is being obtained.

            preprocessor_obj = self.get_data_transformer_object() # Get the preprocessor object by calling the get_data_transformer_object method.

            target_column_name = 'math_score' # Define the target column name.
            numerical_columns = ['writing_score', 'reading_score'] # List of numerical columns in the dataset.

            input_features_train_df = train_df.drop(columns=[target_column_name]) # Drop the target column from the training data to get the input features.
            target_feature_train_df = train_df[target_column_name] # Get the target column from the training data.

            input_features_test_df = test_df.drop(columns=[target_column_name]) # Drop the target column from the testing data to get the input features.
            target_feature_test_df = test_df[target_column_name] # Get the target column from the testing data.

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            ) # Log that preprocessing is about to be applied to both dataframes.

            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_df) # Fit the preprocessor object on the training input features and transform them.
            input_features_test_arr = preprocessor_obj.transform(input_features_test_df) # Transform the testing input features using the fitted preprocessor object (no re-fitting, to avoid data leakage).

            train_arr = np.c_[
                input_features_train_arr, np.array(target_feature_train_df) # Concatenate the transformed training input features and the target column into a single array. c_ is a shorthand for concatenation along the second axis (columns).
                ] 
            test_arr = np.c_[
                input_features_test_arr, np.array(target_feature_test_df) # Concatenate the transformed testing input features and the target column into a single array.
            ]
            
            logging.info(f"Saved preprocessing object.") # Log that the preprocessor object has been saved.

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path, # Specify the file path to save the preprocessor object.
                obj=preprocessor_obj # Specify the preprocessor object to be saved.
            ) # Persist the fitted preprocessor to disk so it can be reused later, e.g. during inference.

            return (
                train_arr, # Transformed training array (features + target).
                test_arr, # Transformed testing array (features + target).
                self.data_transformation_config.preprocessor_obj_file_path # Path where the preprocessor object was saved.
            )
        except Exception as e:
            raise CustomException(e, sys) # Raise a custom exception if any error occurs during the data transformation process.