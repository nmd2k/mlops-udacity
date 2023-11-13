import os
import sys
import logging
import pickle
import pandas as pd

from model.data import process_data
from model.model import inference
sys.path.append(["../"])


# from model.model import inference
# from model.data import process_data


logger = logging.getLogger(__name__)

class InferenceModel:
    "Simple generate function and embedding functions"
    def __init__(self, path="./model/weight") -> None:
        self.cat_features = ["workclass", "education", "marital-status",
            "occupation", "relationship", "race", "sex", "native-country",
        ]
        model_path = os.path.join(path, 'trainedmodel.pkl')
        encoder_path = os.path.join(path, 'encoder.pkl')
        lb_path = os.path.join(path, 'lb.pkl')

        self.model = pickle.load(open(model_path, 'rb'))
        self.encoder = pickle.load(open(encoder_path, 'rb'))
        self.lb = pickle.load(open(lb_path, 'rb'))

    async def predict(self, data):
        """
        Predict function
        """
        try:
            data = {k.replace('_', '-'):[v] for k,v in data.items()}
            input_data = pd.DataFrame.from_dict(data)
            X = process_data(input_data, categorical_features=self.cat_features,
                label=None, training=False, encoder=self.encoder, lb=self.lb)[0]
            
            pred = inference(model = self.model, X = X)
            return {"prediction": "<=50K" if pred[0] == 0 else ">50K"}
        except Exception as exc:
            logger.error(exc)
            
            