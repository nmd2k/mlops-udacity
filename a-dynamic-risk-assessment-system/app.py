import os
import json
import pickle
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np

from scoring import score_model
from diagnostics import model_predictions, dataframe_summary, \
                        missing_data, execution_time, outdated_packages_list


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
deployed_model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    dataset_path = request.form.get('path')
    preds = model_predictions(deployed_model_path, dataset_path[1:-1])
    #add return value for prediction outputs
    return  json.dumps([int(item) for item in preds]) 

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():        
    #check the score of the deployed model
    f1 = score_model(deployed_model_path, test_data_path)
    #add return value (a single F1 score number)
    return json.dumps(f1)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    df_stats = dataframe_summary(dataset_csv_path)
    #return a list of all calculated summary statistics
    return json.dumps(df_stats.to_dict())

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats():        
    #check timing and percent NA values
    time_run = execution_time()
    na_percents = missing_data(dataset_csv_path)
    dependencies = outdated_packages_list().to_dict('records')
    diagnose_dict = {
        'time_run': time_run,
        'na_percents': na_percents,
        'dependencies': dependencies
    }
    #add return value for all diagnostics
    return json.dumps(diagnose_dict)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
