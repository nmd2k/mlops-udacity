# Script to train machine learning model.
import os
import sys
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from data import process_data
from model import train_model, compute_model_metrics, inference

sys.path.append(["../"])

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

import pandas as pd
import pickle
from model import inference, compute_model_metrics
from data import process_data


def model_slicing(model, df, categorical_features, encoder, lb, slice_features):
    """
    Slice model for categorical features
    """

    metrics = {}
    for value in df[slice_features].unique():
        X_slice = df[df[slice_features] == value]
        X_slice, y_slice, _, _ = process_data(X_slice, categorical_features, 
                        label="salary", training=False, encoder=encoder, lb=lb)
        
        preds = inference(model, X_slice)
        logger.info(
            f"shape of preds: {preds.shape} & shape of y_slice: {y_slice.shape}")
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        metrics[value] = {'Precision': precision,
                                'Recall': recall, 'Fbeta': fbeta}
        logger.info(
            f"Slice metrics for {slice_features} = {value}: {metrics[value]}")

    # Write results to slice_output.txt
    with open('./slice_output.txt', 'a') as f:
        for key, value in metrics.items():
            f.write(f"{slice_features} = {key}:{value}")
            f.write("\n")
    return metrics


logger.info("Loading data")

# Add code to load in the data.
data = pd.read_csv("data/cleaned_census.csv")
data.drop(columns=["Unnamed: 0"], inplace=True)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

logger.info("Splitting dataset")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, 
                                    label="salary", training=False, 
                                    encoder=encoder, lb=lb)


logger.info("Start training")
# Train and save a model.
model = train_model(X_train, y_train)

save_path = './model/weight'
os.makedirs(save_path, exist_ok=True)

model_path = os.path.join(save_path, 'trainedmodel.pkl')
encoder_path = os.path.join(save_path, 'encoder.pkl')
lb_path = os.path.join(save_path, 'lb.pkl')

pickle.dump(model, open(model_path, 'wb'))
pickle.dump(encoder, open(encoder_path, 'wb'))
pickle.dump(lb, open(lb_path, 'wb'))

# Evaluate the model
y_preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

logger.info("Precision: %s", precision)
logger.info("Rrecall: %s", recall)
logger.info("Fbeta: %s", fbeta)

for each_cat in cat_features:
    model_slicing(model, df=data, categorical_features=cat_features, 
              encoder=encoder, lb=lb, slice_features=each_cat)
