import sys
import unittest
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.append(["../"])

from model.model import train_model, compute_model_metrics, inference

class TestAPI(unittest.TestCase):
    """Test API utils"""
    def test_train_model(self,):
        """
        Test pipeline of training model
        """
        X = np.random.rand(20, 5)
        y = np.random.randint(2, size=20)
        model = train_model(X, y)
        
        self.assertIsInstance(model, BaseEstimator)
        self.assertIsInstance(model, ClassifierMixin)


    def test_compute_model_metrics(self,):
        """
        Test compute_model_metrics
        """
        y_true, y_preds = [1, 1, 0], [0, 1, 1]
        precision, recall, fbeta = compute_model_metrics(y_true, y_preds)
        for item in [precision, recall, fbeta]:
            self.assertIsNotNone(item)


    def test_inference(self,):
        """
        Test inference of model
        """
        X = np.random.rand(20, 5)
        y = np.random.randint(2, size=20)
        model = train_model(X, y)
        y_preds = inference(model, X)
        
        self.assertEqual(y.shape, y_preds.shape)


if __name__ == '__main__':
    unittest.main()
