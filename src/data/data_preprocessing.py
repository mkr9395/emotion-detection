import pandas as pd
import numpy as np
from pathlib import Path
import os

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

import logging

# logging configuration

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# fetch data from data/raw
def fetch_data(train_data_path:str ,test_data_path:str) -> pd.DataFrame: 
    try:
        train_data  = pd.read_csv(train_data_path)
        test_data  = pd.read_csv(test_data_path)
        
        logger.debug("train and test data fetched from %s and %s",train_data_path, test_data_path)
        
        return train_data, test_data
    except Exception as e:
        logger.error('Data path incorrect %s',e)
        raise

# data cleaning:
def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    try:
        df.content=df.content.apply(lambda content : lower_case(content))
        df.content=df.content.apply(lambda content : remove_stop_words(content))
        df.content=df.content.apply(lambda content : removing_numbers(content))
        df.content=df.content.apply(lambda content : removing_punctuations(content))
        df.content=df.content.apply(lambda content : removing_urls(content))
        df.content=df.content.apply(lambda content : lemmatization(content))
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error('Error during normalization of data', e)
        raise



def store_data(joined_path:str, train_processed_data:pd.DataFrame, test_processed_data:pd.DataFrame) -> None:
    try:
    # store the data inside data/processed
        joined_path.mkdir(parents=True,exist_ok=True)


        train_processed_data.to_csv(joined_path/ "train_processed.csv",index=False)
        test_processed_data.to_csv(joined_path/ "test_processed.csv", index=False)
        logger.debug('processed traina and test data saved %s',joined_path)
    except Exception as e:
        logger.error('Failed to save processed data: %s', e)
        print(f"Error: {e}")
        raise

def main():
    try:
        train_data_path = './data/raw/train.csv'
        test_data_path = './data/raw/test.csv'
        
        train_data, test_data = fetch_data(train_data_path,test_data_path)
        # transform the data

        nltk.download('wordnet')
        nltk.download('stopwords')

        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)
        
        joined_path = Path.joinpath(Path.cwd() / 'data'/'interim')
        
        store_data(joined_path, train_processed_data, test_processed_data)
        
        logger.debug('processed data stored to processed folder : %s', joined_path)
    
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}") 
        raise
        

if __name__ == '__main__':
    main()