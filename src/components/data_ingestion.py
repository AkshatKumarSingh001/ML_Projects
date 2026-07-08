''' This data ingestion file is responsible for ingesting data from various sources and preparing it for further processing.
It includes functions to read data from files, databases, and APIs, as well as functions to clean and transform the data into a suitable format for analysis. 
The module also handles error checking and logging to ensure that the data ingestion process is robust and reliable.
NOTE: After data ingestion, data transformation happens in the data_transformation.py file. 
'''

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'notebook', 'data', 'stud.csv')

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(PROJECT_ROOT, 'artifacts', 'train.csv') # Artifacts folder is created to store the train and test data after ingestion and train.csv is the name of the train data file.
    test_data_path: str = os.path.join(PROJECT_ROOT, 'artifacts', 'test.csv') # test.csv is the name of the test data file.
    raw_data_path: str = os.path.join(PROJECT_ROOT, 'artifacts', 'data.csv') # data.csv is the name of the raw data file.

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # Create an instance of the DataIngestionConfig class to access the file paths.

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component") # Log the start of the data ingestion process.
        
        try:
            df = pd.read_csv(RAW_DATA_PATH) # Read the raw data from a CSV file into a pandas DataFrame. 
            # NOTE: You can just make changes in the above line to read data from any other source like database or API. Just make sure to change the path accordingly.

            logging.info("Read the dataset as dataframe") # Log that the dataset has been read successfully.

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # Create the artifacts directory if it doesn't exist.

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) # Save the raw data to a CSV file in the artifacts directory.
            logging.info("Train test split initiated") # Log that the train-test split is about to begin.

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) # Split the data into training and testing sets (80% train, 20% test).

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) # Save the training set to a CSV file.
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) # Save the testing set to a CSV file.

            logging.info("Ingestion of the data is completed") # Log that the data ingestion process is complete.

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            ) # Return the paths to the training and testing data files.

        except Exception as e:
            raise CustomException(e, sys) # Raise a custom exception if an error occurs during data ingestion.
        

if __name__ == "__main__":
    obj = DataIngestion() # Create an instance of the DataIngestion class.
    train_data,test_data = obj.initiate_data_ingestion() # Call the initiate_data_ingestion method to perform data ingestion and get the paths to the training and testing data files.

    data_transformation = DataTransformation() # Create an instance of the DataTransformation class.
    data_transformation.initiate_data_transformation(train_data, test_data) # Call the initiate_data_transformation method to perform data transformation.

