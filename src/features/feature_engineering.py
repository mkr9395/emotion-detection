import pandas as pd
import numpy as np
from pathlib import Path
import os

import yaml

from sklearn.feature_extraction.text import CountVectorizer

import logging


# loading from yaml file
def load_params(params_path:str) -> float:
    
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)

    max_features = params['feature_engineering']['max_features']
    
    return max_features
    


# fetch data from data/processed
def fetch_data(train_data_path:str, test_data_path:str) -> (pd.DataFrame, pd.DataFrame):

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    return train_data, test_data

# replacing the na values : 
def fill_na_values(train_data:pd.DataFrame, test_data:pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    train_data.fillna('',inplace=True)
    test_data.fillna('',inplace=True)
    
    return train_data, test_data


# dividing data into X_train and X_test as BOG works only on X

def split_train_test(train_data:pd.DataFrame,test_data:pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    
    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values

    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values
    
    return X_train, X_test, y_train, y_test


def feature_engineering(vectorizer:CountVectorizer, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> (pd.DataFrame, pd.DataFrame):
    
    # Fit the vectorizer on the training data and transform it
    X_train_bow = vectorizer.fit_transform(X_train)
    # Transform the test data using the same vectorizer
    X_test_bow = vectorizer.transform(X_test)
    
    # joining y_train and X_train
    train_df = pd.DataFrame(X_train_bow.toarray())
    train_df['label'] = y_train

    # joining y_test and X_test
    test_df = pd.DataFrame(X_test_bow.toarray())
    test_df['label'] = y_test   
    
    return train_df , test_df 

# save data after feature engineering in data/features folder
def save_data(joined_path:str, train_df:pd.DataFrame, test_df:pd.DataFrame)-> None:
        
    # store the data inside data/features
    joined_path.mkdir(parents=True,exist_ok=True)


    train_df.to_csv(joined_path/ "train_bow.csv",index=False)
    test_df.to_csv(joined_path/ "test_bow.csv", index=False)


def main():
    
    max_features = load_params('params.yaml')
    
    train_data_path = 'data/interim/train_processed.csv'
    test_data_path = 'data/interim/test_processed.csv'
    
    train_data, test_data = fetch_data(train_data_path,test_data_path)
    
    train_data, test_data = fill_na_values(train_data, test_data)
    
    X_train, X_test, y_train, y_test = split_train_test(train_data, test_data)
    
    # Apply Bag of Words (CountVectorizer)
    vectorizer = CountVectorizer(max_features=max_features)
    
    train_df , test_df = feature_engineering(vectorizer, X_train, X_test,y_train, y_test)
    
    joined_path = Path.joinpath(Path.cwd() / 'data'/'processed')
    
    save_data(joined_path, train_df , test_df)
    
    
if __name__ == '__main__':
    
    main()    
    