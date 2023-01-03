"""Unit test of main.py API module"""

from fastapi.testclient import TestClient
#from fastapi import HTTPException
import json
import logging
from starter.main import app
client = TestClient(app)

logging.basicConfig(filename='test_logging.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')


def test_root():
    """
    Test welcome message for get at root
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to our model API"


def test_inference():
    """
    Test model inference output
    """
    sample =  {  'age':38,
                'workclass':"Self-emp-inc", 
                'fnlgt':99146,
                'education':"Bachelors",
                'education_num':13,
                'marital_status':"Married-civ-spouse",
                'occupation':"Exec-managerial",
                'relationship':"Husband",
                'race':"White",
                'sex':"Male",
                'capital_gain':15024,
                'capital_loss':0,
                'hours_per_week':80,
                'native_country':"United-States"
            }

    data = json.dumps(sample)

    r = client.post("/inference/", data=data )

    # test response and output
    assert r.status_code == 200
    assert r.json()["age"] == sample['age']
    assert r.json()["fnlgt"] == sample['fnlgt']

    # test prediction vs expected label
    logging.info(f'********* prediction = {r.json()["prediction"]} ********')
    assert r.json()["prediction"] == '>50K'


def test_inference_class():
    """
    Test model inference output for class <=50k
    """
    sample =  {  'age':30,
                'workclass':"Private", 
                'fnlgt':234721,
                'education':"HS-grad",
                'education_num':1,
                'marital_status':"Separated",
                'occupation':"Handlers-cleaners",
                'relationship':"Not-in-family",
                'race':"Black",
                'sex':"Male",
                'capital_gain':0,
                'capital_loss':0,
                'hours_per_week':35,
                'native_country':"United-States"
            }

    data = json.dumps(sample)

    r = client.post("/inference/", data=data )

    # test response and output
    assert r.status_code == 200
    assert r.json()["age"] == 30
    assert r.json()["fnlgt"] == 234721

    # test prediction vs expected label
    logging.info(f'prediction = {r.json()["prediction"]}')
    assert r.json()["prediction"][0] == '<=50K'
        
    
if '__name__' == '__main__':
    test_root()
    test_inference()
    test_inference_class()