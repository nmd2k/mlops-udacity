import os
import json
import pickle
from glob import glob

import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model = os.path.join(config['output_model_path'])

model = pickle.load(open(f'{output_model}/trainedmodel.pkl', 'rb'))


##############Function for reporting
def score_model(model):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    df = pd.read_csv(glob(f'{test_data_path}/*.csv')[0])
    X_test = df.drop(['corporation', 'exited'], axis='columns')
    y_test = df['exited']
    
    predictions = model.predict(X_test)
    
    cf_matrix = metrics.confusion_matrix(y_test, predictions)
    
    sns.heatmap(cf_matrix, annot=True, fmt='d', cbar=False, cmap='icefire')
    plt.savefig(os.path.join(config["output_model_path"], "confusionmatrix2.png"))


if __name__ == '__main__':
    score_model(model,)
