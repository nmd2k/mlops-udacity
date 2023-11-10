import os
import json
import subprocess
import logging


import training
import scoring
import deployment
import diagnostics
import reporting

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='./logs/scoring.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']

test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
dataset_csv_path = os.path.join(output_folder_path, 'finaldata.csv')

ingested_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
latest_score = os.path.join(prod_deployment_path, 'latestscore.txt')

##################Check and read new data
#first, read ingestedfiles.txt
with open(ingested_path) as f:
    ingested_files_list = f.read().splitlines()

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
all_files_exist = True
for file_name in os.listdir(input_folder_path):
    file_path = os.path.join(input_folder_path, file_name)
    if file_path not in ingested_files_list:
        all_files_exist = False
        break

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if all_files_exist:
    logger.info("No new data found. Exit the process")
    exit()
else:
    logger.info("New data available %s", input_folder_path)
    subprocess.call(['python', 'ingestion.py'])

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(latest_score, 'r') as f:
    latest_score = float(f.read())
new_score = scoring.score_model(model_path, dataset_csv_path)
logger.info(f'Lastest F1 score: {latest_score}, F1 score on newly ingested data: {new_score}')

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if new_score >= latest_score:
    logger.info('No model drift')
    exit()
else:
    logger.info('Found model drift. Prepare to re-train model')

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
subprocess.call(['python', 'training.py'])
subprocess.call(['python', 'scoring.py'])
subprocess.call(['python', 'deployment.py'])

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
subprocess.call(['python', 'diagnostics.py'])
subprocess.call(['python', 'reporting.py'])
subprocess.call(['python', 'apicalls.py'])
