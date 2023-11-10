import os
import json
import logging
import pickle
from glob import glob
import pandas as pd

from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='./logs/scoring.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])

model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')


#################Function for model scoring
def score_model(model_path, test_data_path):
    # this function should take a trained model, load test data, and calculate an 
    # F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file
    logger.info("Load model from: %s", model_path)
    model = pickle.load(open(model_path, 'rb'))
    
    logger.info("Load data from: %s", test_data_path)
    df = pd.read_csv(test_data_path)  # assert only 1 file in test
    X_test = df.drop(['corporation', 'exited'], axis='columns')
    y_test = df['exited']
    
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)

    score_saving_path = config['output_model_path']
    os.makedirs(score_saving_path, exist_ok=True)
    
    with open(os.path.join(score_saving_path, "latestscore.txt"), 'w') as f:
        f.write(str(f1))
    logger.info("Score saved!")
    
    return f1


if __name__ == "__main__":
    score_model(model_path, test_data_path=glob(f'{test_data_path}/*.csv')[0])
