import os
import logging
import pickle
import json
import glob

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='./logs/training.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 

def load_data(data_path):
    list_files = glob.glob(f'{data_path}/*.csv')
    # Load the concaternated dataset
    assert len(list_files) == 1
    
    df = pd.read_csv(list_files[0])
    X = df.drop(['corporation', 'exited'], axis='columns')
    y = df['exited']
    
    return X, y


#################Function for training the model
def train_model():
    logger.info("Loading data ...")
    X_train, y_train = load_data(dataset_csv_path)
    
    logger.info("Initualized model ...")
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    logger.info("Training model ...")
    model.fit(X_train, y_train)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    save_path = os.path.join(model_path, "trainedmodel.pkl")
    
    os.makedirs(config['output_model_path'], exist_ok=True)
    pickle.dump(model, open(save_path, 'wb'))
    logger.info("Model saved %s", config['output_model_path'])


if __name__ == "__main__":
    train_model()
