from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import logging


logging.basicConfig(filename='logging.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    logging.info("Start training RandomForestClassifier")

    clf = RandomForestClassifier(max_depth=2, random_state=0)

    clf.fit(X_train, y_train)
    logging.info("********* Finish training! ***********")

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds


def compute_slices(df, feature, y, preds):
    """
    Compute the performance on slices for a given categorical feature
    ------
    df: 
        test dataframe
    feature:
        feature on which to perform the slices
    y : np.array
        True labels
    preds : np.array
        Predicted labels
    Returns
    ------
    Dataframe with
        n_samples: integer - number of data samples in the slice
        precision : float
        recall : float
        fbeta : float
    row corresponding to each of the unique values taken by the feature (slice)
    """    
    slice_values = df[feature].unique().tolist()
    res_df = pd.DataFrame(index=slice_values, 
                          columns=['feature','n_samples','precision', 'recall', 'fbeta'])
    for val in slice_values:
        slice_y = y[df[feature]==val]
        slice_preds = preds[df[feature]==val]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)
        
        res_df.at[val, 'feature'] = feature
        res_df.at[val, 'n_samples'] = len(slice_y)
        res_df.at[val, 'precision'] = precision
        res_df.at[val, 'recall'] = recall
        res_df.at[val, 'fbeta'] = fbeta

    res_df.reset_index(names='feature value', inplace=True)

    return res_df
