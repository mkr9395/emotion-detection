import pandas as pd
import numpy as np
import pathlib
import os
from pathlib import Path


import xgboost as xgb

import pickle

import yaml


# loading from yaml file
def load_params(params_url: str) -> (int, float):
    with open(params_url, 'r') as file:
        params = yaml.safe_load(file)

    n_estimators = params['model_building']['n_estimators']
    learning_rate = params['model_building']['learning_rate']
    
    return n_estimators, learning_rate


# reading the data from processed folder:

def load_data(data_path:str) -> (pd.DataFrame, pd.Series):
    train_data = pd.read_csv(data_path)

    X_train = train_data.iloc[:,:-1]
    y_train = train_data.iloc[:,-1]
    
    return X_train, y_train
    
    
# Build the model
def build_model(n_estimators: int, learning_rate: float) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', learning_rate=learning_rate, n_estimators=n_estimators)
    return model

# Train the model
def train_model(model: xgb.XGBClassifier, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    model.fit(X_train, y_train)
    return model


# pickling the model
def pickling(model_location:str,fitted_model) -> None:
    pickle.dump(fitted_model,open(model_location,'wb')) # dumping in write binary mode


def main():
    
    n_estimators, learning_rate = load_params('params.yaml')
    
    data_path = 'data/features/train_bow.csv'
    
    X_train, y_train = load_data(data_path)
    
    # Build and train the model
    model = build_model(n_estimators, learning_rate)
    fitted_model = train_model(model, X_train, y_train)
    
    
    model_location = 'model.pkl'
    
    pickling(model_location,fitted_model)
    

if __name__ == '__main__':
    main() 