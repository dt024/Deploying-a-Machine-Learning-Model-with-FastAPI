# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Prediction task is to classify whether a person makes over 50K a year. In this project RandomForestClassifier is used within scikit-learn 1.2.0. Parameters used are:

- learning_rate: 1.0
- max_depth: 2
- min_samples_split: 2
- random_state: 0 
- min_samples_leaf: 1
- bootstrap: True
- n_estimators: 100 

Model is saved as a pickle file in the model folder. All training steps and metrics are logged in the file "logging.log".

## Intended Use
This model can be used to predict the salary level of an individual.

## Training Data
The Census Income Dataset was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). The original csv file has 32.561 rows and 15 columns including binary target label "salary", 8 categorical features and 6 numerical features. Please check the above link for more detail.
Target label "salary" has two classes ('<=50K', '>50K') with a ratio of circa 75% / 25% which is highly imbalance.
A simple data cleansing was performed on the original dataset to remove leading and trailing whitespaces. See EDA_cleaning.ipynb notebook for cleansing step.

The dataset was split into a train set and a test set using an 80-20 split with salary-based stratification.
A label binarizer was applied to the target and a One Hot Encoder was applied to the category features to use the data for training.

## Evaluation Data
20% of the dataset was used for model evaluation.
Same pre-processing as Training Data is applied to Evaluation Data.

## Metrics
The classification performance is evaluated using precision, recall and fbeta metrics.

The model achieves below scores using the test set:
- precision:0.759
- recall:0.643
- fbeta:0.696

## Ethical Considerations
The dataset shouldn't be regarded as an accurate representation of the pay distribution or as a basis for assuming the salary level of particular population categories.

## Caveats and Recommendations
The 1994 Census database was mined for information. As an out-of-date sample, the dataset is insufficient for use as a statistical representation of the population. It is advised to utilize the dataset to train on ML classification or other related issues.