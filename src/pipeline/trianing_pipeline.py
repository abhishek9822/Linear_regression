from src.components.data_transformation import DataTransformation
import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer
from src import exception
from src import logger
import pandas as pd

if __name__ =='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)