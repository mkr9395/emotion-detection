import pandas as pd
import numpy as np

import pathlib
from pathlib import Path

import pickle
import json

import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report

# loading the test data from processed folder:
def load_data(data_path:str) -> (pd.DataFrame, pd.DataFrame):

    test_data = pd.read_csv(data_path)

    X_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:,-1]
    
    return X_test, y_test


# fetching the model
def load_model(model_path:str):
    with open(model_path, 'rb')  as f:
        model = pickle.load(f)    
        
    return model

# model prediction:
def model_prediction(model,X_test:pd.DataFrame)-> (np.ndarray, np.ndarray):
    # making prediction on test data : 
    y_pred = model.predict(X_test)

    # using predict proba
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return y_pred, y_pred_proba


# all metrics
def get_metrics(y_test: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> (float, float, float, float, str):
    
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    classification_rep = classification_report(y_test, y_pred)

    return accuracy, precision, recall, auc, classification_rep

# putting metrics in dictionary format
def create_metrics_dict(accuracy, precision, recall, auc ):
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc' : auc    
    }
    return metrics_dict


# converting dictionary in json file

def save_metrics_to_json(metrics_dict: dict, json_path: str) -> None:
    with open(json_path, 'w') as file:
        json.dump(metrics_dict, file, indent=4)

    

def main():
    
    data_path = 'data/processed/test_bow.csv'
    
    X_test, y_test = load_data(data_path)
    
    model_path = 'models/model.pkl'
    model = load_model(model_path)
    
    y_pred, y_pred_proba = model_prediction(model,X_test)
    
    accuracy, precision, recall, auc, classification_rep = get_metrics(y_test, y_pred, y_pred_proba)
    
    metrics_dict = create_metrics_dict(accuracy, precision, recall, auc)
    json_path = 'metrics.json'
    save_metrics_to_json(metrics_dict, json_path)
    
if __name__ == '__main__':
    main() 