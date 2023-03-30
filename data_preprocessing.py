import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def import_dataset(file, start):
    dataset = pd.read_csv(file)
    X = dataset.iloc[:, start:-1].values     # Matrix of the features
    y = dataset.iloc[:, -1].values      # Dependent variable
    return X, y

def missing_data(X, strategy):
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputer.fit(X[:, 1:3])  # Works on numerical values only.  All rows, only columns 1 & 2 (Age & Salary)
    X[:, 1:3] = imputer.transform(X[:, 1:3])    # fit connects imputer to matrix; transform applies the changes and returns
    return X

# CONVERTS STRINGS INTO NUMBERS USING ONE HOT ENCODING (MULTIPLE COLUMNS...BINARY VECTORS)
def encode_categorical_data_independent_variable(X, field):
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), field)], remainder='passthrough')
    # transformer says encode using OnHotEncoder on [0] column(s) and remainder says passthrough the other columns
    X = np.array(ct.fit_transform(X))   # np.array converts the returned value into numpy array that will be needed later
    return X

# CONVERTS YES/NO INTO 0s and 1s
def encode_categorical_data_dependent_variable(y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    return y

def split_into_training_and_test_sets(X, y, rs):
    return train_test_split(X, y, test_size=0.2, random_state=rs)

def feature_scaling(X_train, X_test):
    sc = StandardScaler()
    # DO NOT NEED TO APPLY TO DUMMY VARIABLES FROM ONEHOTENCODER
    X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
    X_test[:, 3:] = sc.transform(X_test[:, 3:])
    return X_train, X_test

def data_preprocessing(file, addMissing, encodeCategoricalX, encodeCategoricalY, field, split, start):
    # IMPORT THE DATASET
    X, y = import_dataset(file, start)

    # MISSING DATA-REPLACE MISSING DATA WITH A STRATEGY
    if addMissing:
        X = missing_data(X, 'mean')

    if encodeCategoricalX:
        # ENCODING CATEGORICAL DATA
        X = encode_categorical_data_independent_variable(X, field)

    if encodeCategoricalY:
        y = encode_categorical_data_dependent_variable(y)

    if split:
        # SPLIT INTO TRAINING AND TEST SETS
        X_train, X_test, y_train, y_test = split_into_training_and_test_sets(X, y, 0)

        # FEATURE SCALING (AVOIDS SOME FEATURES NOT TO DOMINATE OTHER FEATURES)
        # NOT DONE ON ALL TYPES I.E. NOT DONE ON REGRESSION
        # NOT CALLING IN THIS FUNCTION BUT RATHER OUTSIDE OF FUNCTION

        return X_train, X_test, y_train, y_test
    else:

        return X, y