import os
import json
import pickle
import logging
import subprocess
from time import time
from glob import glob

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='./logs/diagnostics.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')

##################Function to get model predictions
def model_predictions(deployed_model_path, test_data_path):
    #read the deployed model and a test dataset, calculate predictions
    model = pickle.load(open(deployed_model_path, 'rb'))
    df_test = pd.read_csv(test_data_path)
    
    X_test = df_test.drop(['corporation', 'exited'], axis='columns')
    preds = model.predict(X_test)
    
    #return value should be a list containing all predictions
    logger.info(preds)
    return preds

##################Function to get summary statistics
def dataframe_summary(data_path):
    #calculate summary statistics here
    df = pd.read_csv(data_path)
    df_stats = df.describe().iloc[1:3]
    median_list = []
    
    for col in df_stats.columns:
        median_list.append(df[col].median(axis=0))
    
    df_median = pd.DataFrame([median_list], columns=df_stats.columns, index=['median'])
    df_stats = pd.concat([df_stats, df_median])
    
    #return value should be a list containing all summary statistics
    logger.info(df_stats)
    return df_stats


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    exec_time = []
    start = time()
    os.system('python3 ingestion.py')
    exec_time.append(time() - start)
    logger.info("Execution time of ingestion.py: %s", exec_time[0])
    
    start = time()
    os.system('python3 training.py')
    exec_time.append(time() - start)
    logger.info("Execution time of training.py: %s", exec_time[1])
    
    #return a list of 2 timing values in seconds
    return exec_time[0], exec_time[1]

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    outdated = subprocess.check_output(['pip', 'list','--outdated'])
    with open('requirements.txt', 'wb') as f:
       f.write(outdated)
       
    logger.info(outdated)
    return outdated     

def missing_data(data_path):
    df = pd.read_csv(data_path)
    
    na_list = list(df.isna().sum(axis=0))
    na_percents = [na_list[i]/len(df.index) for i in range(len(na_list))]
    
    logger.info(na_percents)
    return na_percents



if __name__ == '__main__':
    model_predictions(prod_deployment_path, glob(f'{test_data_path}/*.csv')[0])
    dataframe_summary(glob(f'{dataset_csv_path}/*.csv')[0])
    missing_data(glob(f'{dataset_csv_path}/*.csv')[0])
    execution_time()
    outdated_packages_list()

