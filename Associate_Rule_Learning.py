import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

# APRIORI IS MUCH BETTER THAN ECLAT


def apriori_func(file):
    dataset = pd.read_csv(file, header=None)
    transactions = []
    for i in range(len(dataset)):
        transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

    # Training Apriori model on dataset
    rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
    # Calculating min_support...say item appears in 3 transactions per day * 7 days/week divided by total # of transactions
    # min_confidence...trial and error (start at 0.8 and see how many rules, divide by 2 each time to get best fit)
    # min_lift...should be at least 3....
    # min_length...since we're doing buy 1 of type A get 1 of type A should only have 2 items
    # max_length....see min_length

    # Visualizing the rules
    results = list(rules)
    results_in_dataframe = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side', 'Support',
                                                                   'Confidence', 'Lift'])
    results_in_dataframe = results_in_dataframe.nlargest(n=10, columns='Lift')
    print(results_in_dataframe)

def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))