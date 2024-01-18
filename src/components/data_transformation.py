'''Data transformation-
It is the process of converting, cleansing, and structuring data
into a usable format.'''

import os
import sys
from dataclasses import dataclass

import pandas as pd 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


'''
  # One Hot Encoding- One hot encoding is a technique that we use to represent categorical variables as numerical values in a ML model.

  # ColumnTransformer- It is a class in scikit-learn that allows you to apply different transformers to different subsets of columns in your dataset.
It is particularly useful when you have a dataset with columns that require different preprocessing steps.
This transformer applies the specified transformers to the specified columns and concatenates the results.

  # SimpleImputer- It is a class in scikit-learn used for imputing missing values in your dataset.
It provides simple strategies for imputation, such as replacing missing values with the mean, median, or most frequent values of the respective columns.

  # StandardScaler- is a class in scikit-learn that provides a method for standardizing features by removing the mean and scaling to unit variance.
such as gradient-based optimization algorithms. 

                    Original value - Mean
Standardized value= ----------------------
                      Standard deviation
 '''

# @dataclass- It is a class that is designed to only hold data values.
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for creating the data transformer object.
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            # Numeric pipeline for numerical columns
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )
            # Categorical pipeline for categorical columns
            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")), # missing_values -most_freq = mode
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # ColumnTransformer to apply different transformations to numerical and categorical columns
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
             # Log and raise an exception with custom information
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # Get the data transformer object
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input and target features for training data
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            # Separate input and target features for testing data
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Apply the data transformer to the training and testing data
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # Combine input features and target features
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # Save the preprocessing object to a file
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)