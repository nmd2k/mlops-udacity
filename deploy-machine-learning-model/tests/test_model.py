import os
import sys
import json
import unittest
import requests

sys.path.append(["../"])
URL = os.getenv("HOST_URL")


def request(data):
    response = requests.post(URL+"predict", json = data)
    return response


class TestModel(unittest.TestCase):
    """Test model utils"""
    def test_root(self):
        """Test"""
        r = requests.get(URL)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["message"], "Hello, welcome to our app!")

    def test_predict_positive(self):
        """Test"""
        data = {"age": 52,
                "workclass": "Self-emp-inc",
                "fnlgt": 287927,
                "education": "HS-grad",
                "education_num": 9,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Wife",
                "race": "White",
                "sex": "Female",
                "capital_gain": 15024,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
                }
        response = request(data=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"prediction": ">50K"})

    def test_predict_negative(self,):
        """Test"""
        data = {"age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
                }
        response = request(data=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"prediction": "<=50K"})


if __name__ == "__main__":
    unittest.main()