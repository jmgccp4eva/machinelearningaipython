import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def ann(file, start, gender_column, num_hidden_layers):
    # PART 1: DATA PREPROCESSING

    dataset = pd.read_csv(file)
    X = dataset.iloc[:, start:-1].values
    y = dataset.iloc[:, -1].values

    # Encoding of categorical data
    le = LabelEncoder()
    X[:, gender_column] = le.fit_transform(X[:, gender_column])

    # One Hot Encoding Geography
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # PART 2: BUILING AN ANN

    # Initializing the ANN
    an_network = tf.keras.models.Sequential()

    # Adding layers
    for _ in range(num_hidden_layers):
        an_network.add(tf.keras.layers.Dense(units=6, activation='relu'))   # 6 hidden neurons, relu is rectifier function codename

    # Adding the Outer Layer
    an_network.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    # When doing binary, must use sigmoid.  when doing non-binary classification activation should be softmax

    # PART 3: TRAINING AN ANN

    # Compiling the ANN
    an_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # When doing binary must use binary_crossentropy.  When doing non-binary must use categorical_crossentropy for loss

    # Training the ANN on the Training Set
    an_network.fit(X_train, y_train, batch_size=32, epochs=100)

    # PART 4: MAKING PREDICTIONS AND EVALUATION OF MODEL

    # Predicting France, M, 40y/o, Tenure: 3 years, Balance $60K, Number of products: 2, Have CC: Yes, Active Member: Yes
    # Estimated salary: $50K, should we say goodbye?
    # print(an_network.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

    # Predicting the Test Results
    y_pred = an_network.predict(X_test)
    y_pred = (y_pred > 0.5)     # Converts to 0 & 1 or True and False technically
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)