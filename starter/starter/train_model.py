# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pickle, os
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_slices
import logging


logging.basicConfig(filename='loggin.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

# Add code to load in the data.
datapath = "../data/census.csv"
data = pd.read_csv(datapath)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split( data, 
                                test_size=0.20, 
                                random_state=2023, 
                                stratify=data['salary']
                                )

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

savepath = '../model'
filename = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']

# Train and save a model.
model = train_model(X_train, y_train)
# save model  to disk in ./model folder
pickle.dump(model, open(os.path.join(savepath,filename[0]), 'wb'))
pickle.dump(encoder, open(os.path.join(savepath,filename[1]), 'wb'))
pickle.dump(lb, open(os.path.join(savepath,filename[2]), 'wb'))
logging.info(f"Model saved to disk: {savepath}")
logging.info(f"Training params: {model.get_params()}")

# evaluate trained model on test set
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

logging.info(f"precision:{precision:.3f}, recall:{recall:.3f}, fbeta:{fbeta:.3f}")


# Compute performance on slices for categorical features
for feature in cat_features:
    performance_df = compute_slices(test, feature, y_test, preds)
    logging.info(f"Performance on slice {feature}")
    logging.info(performance_df)
