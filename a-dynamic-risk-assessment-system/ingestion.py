import os
import json
import glob
import logging

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='./logs/ingestion.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe() -> None:
    #check for datasets, compile them together, and write to an output file
    df_list = []
    visited = []
    for file in glob.glob(f'{input_folder_path}/*.csv'):
        visited.append(file)
        df_list.append(pd.read_csv(file))
        
    dataframe = pd.concat(df_list, axis=0, ignore_index=False).drop_duplicates()
    
    dataframe.to_csv(os.path.join(output_folder_path, "finaldata.csv"), index=False)

    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as writer:
        for item in visited:
            writer.write(item + "\n")

    logging.info('Datasets found, compiled, and duplicates dropped')  


if __name__ == '__main__':
    merge_multiple_dataframe()
