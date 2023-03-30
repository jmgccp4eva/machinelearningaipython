import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

def natural_language_processing(file):
    dataset = pd.read_csv(file, delimiter='\t', quoting=3)

    nltk.download('stopwords')
    # Cleaning text
    corpus = []
    for i in range(0, len(dataset)):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        with open('stopwords_to_remove.tsv', 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
        for word in lines:
            all_stopwords.remove(word.strip())
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    # Create Bag-of-Words Model
    cv = CountVectorizer()
    X = cv.fit_transform(corpus).toarray()
    print(len(X[0]))
    y = dataset.iloc[:, -1].values

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Training Naive Bayes model on training set
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the test set results
    y_pred = classifier.predict(X_test)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)