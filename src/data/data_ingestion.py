import pandas as pd
import numpy as np

import os
import pathlib
from pathlib import Path

from sklearn.model_selection import train_test_split

import yaml

import logging

# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# function to load params from yaml file
def load_params(params_path:str) -> float:
    """load paramters from yaml file"""
    
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        test_size = params['data_ingestion']['test_size']
        
        logger.info('Paramters retrived from %s',params_path)
        return test_size
    except FileNotFoundError:
        logger.error('File not found: %s',params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
    


# reading the data from url
def read_data(url:str) -> pd.DataFrame :
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(url)
        logger.info('Data loaded from %s', url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def process_data(df:pd.DataFrame)-> pd.DataFrame:
    """Preprocess the data."""
    try:
    
        # dropping tweet column
        df.drop(columns=['tweet_id'],inplace=True)
        logger.info('tweet_id column dropped')
        # dropping rows with na values
        df.dropna(inplace=True)
        logger.info('na values removed')
        # keeping only happiness and sadness columns
        final_df = df[df['sentiment'].isin(['happiness','sadness'])]
        
        logger.info('keeping only happiness and sadness columns')

        # category encoding in output column
        final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)   
        
        logger.info('data preprocessing process complete')
        
        return final_df
    except KeyError as e:
        logger.error('Missing column from dataframe',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred: %s', e)
        raise


# splitting data into test and train:
def data_split(df:pd.DataFrame, test_size:float) -> pd.DataFrame:
    """ splitting data into test and train:"""
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)  
        logger.info('Data has been splitted into train and test')
        return train_data, test_data
    except Exception as e:
        logger.error('Unexpected error occurred: %s', e)
        raise
    

def save_data(joined_path:str, train_data:pd.DataFrame, test_data:pd.DataFrame)-> None:
    """Saving train and test data inside raw folder"""
    try:
        
        joined_path.mkdir(parents=True,exist_ok=True)
    
        train_data.to_csv(joined_path/ "train.csv",index=False)
        test_data.to_csv(joined_path/ "test.csv", index=False)
        logger.info('Train and Test data has been saved in data/raw folder')
    except Exception as e:
        logger.error('Unexpected error occurred: %s', e)
        raise

def main():
    try:
    
        # fetching test size from params.yaml file
        test_size = load_params('params.yaml')
        
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        
        df = process_data(df)
        
        train_data, test_data = data_split(df,test_size)
        
        # creating data folder for raw files
        joined_path = Path.joinpath(Path.cwd() / 'data'/'raw')
        
        save_data(joined_path, train_data, test_data)
        
        logger.info("DATA INGESTION PROCESS COMPLETED")
        
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")
    
if __name__ == '__main__':
    main()    
    
    
    
    
    
    
    


















