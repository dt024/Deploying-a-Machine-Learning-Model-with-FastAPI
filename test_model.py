"Unit tests for model and training/inference processes"

import pytest, os, logging, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

from starter.starter.ml.model import inference, compute_model_metrics
from starter.starter.ml.data import process_data

logging.basicConfig(filename='test_logging.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')


"""
Fixture - The test functions will 
use the return of data() as an argument
"""
@pytest.fixture(scope="module")
def data():
    # code to load in the data.
    datapath = "./starter/data/census.csv"
    return pd.read_csv(datapath)


@pytest.fixture(scope="module")
def path():
    return "./starter/data/census.csv"


@pytest.fixture(scope="module")
def features():
    """
    Fixture - return the categorical features as argument
    """
    cat_features = [    "workclass",
                        "education",
                        "marital-status",
                        "occupation",
                        "relationship",
                        "race",
                        "sex",
                        "native-country"]
    return cat_features


@pytest.fixture(scope="module")
def train_dataset(data, features):
    """
    Fixture - returns train dataset to be used for model testing
    """
    train, test = train_test_split( data, 
                                test_size=0.20, 
                                random_state=10, 
                                stratify=data['salary']
                                )
    X_train, y_train, encoder, lb = process_data(
                                            train,
                                            categorical_features=features,
                                            label="salary",
                                            training=True
                                        )
    return X_train, y_train


def test_import_data(path):
    """
    Test presence and shape of dataset file
    """
    try:
        df = pd.read_csv(path)

    except FileNotFoundError as err:
        logging.error("File not found")
        raise err

    # Check the df shape
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError as err:
        logging.error(
        "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_features(data, features):
    """
    Check that categorical features are in dataset
    """
    try:
        assert sorted(set(data.columns).intersection(features)) == sorted(features)
    except AssertionError as err:
        logging.error(
        "Features are missing in the data columns")
        raise err


def test_is_model():
    """
    Check saved model is present
    """
    path = "./starter/model/trained_model.pkl"
    if os.path.isfile(path):
        try:
            _ = pickle.load(open(path, 'rb'))
        except Exception as err:
            logging.error(
            "Saved model not found!")
            raise err
    else:
        pass


def test_inference(train_dataset):
    """
    Check inference function
    """
    X_train, y_train = train_dataset

    path = "./starter/model/trained_model.pkl"
    if os.path.isfile(path):
        model = pickle.load(open(path, 'rb'))

        try:
            preds = inference(model, X_train)
        except Exception as err:
            logging.error(
            "Inference cannot be performed on saved model and train data")
            raise err
    else:
        pass


def test_compute_model_metrics(train_dataset):
    """
    Check calculation of performance metrics function
    """
    X_train, y_train = train_dataset

    path = "./starter/model/trained_model.pkl"
    if os.path.isfile(path):
        model = pickle.load(open(path, 'rb'))
        preds = inference(model, X_train)

        try:
            precision, recall, fbeta = compute_model_metrics(y_train, preds)
        except Exception as err:
            logging.error(
            "Performance metrics cannot be calculated on train data")
            raise err
    else:
        pass