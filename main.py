from Regression import *
from Classification import *
from Clustering import *
from Associate_Rule_Learning import *
from Reinforcement_Learning import *
from NLP import *
from Artificial_Neural_Network import *
from Convolutional_Neural_Network import *

def main():
    # simple_linear_regression('Data Directory/simple_linear_regression.csv', 0)
    # multi_linear_regression('Data Directory/multiple_linear_regression.csv', [3], 0)
    # polynomial_regression('Data Directory/polynomial_regression.csv', 1)
    # linear_support_vector_regression('Data Directory/support_vector_regression.csv', 1)
    # decision_tree_regression('Data Directory/decision_tree_regression.csv', 1)
    # random_forest_regression('Data Directory/random_forest.csv', 1)

    # logistic_regression('Data Directory/logistic_regression.csv', 0)
    # k_nearest_neighbors('Data Directory/logistic_regression.csv', 0)
    # support_vector_machine('Data Directory/logistic_regression.csv', 0)
    # nonlinear_support_vector_regression('Data Directory/logistic_regression.csv', 0)
    # naive_bayes('Data Directory/logistic_regression.csv', 0)
    # decision_tree_classification('Data Directory/logistic_regression.csv', 0)
    # random_forest_classification('Data Directory/logistic_regression.csv', 0)

    # k_means_clustering('Data Directory/clustering.csv', 3)
    # hierarchical_clustering('Data Directory/clustering.csv', 3)

    # apriori_func('Data Directory/association_rule_learning.csv')

    # upper_confidence_bound('Data Directory/reinforcement_learning.csv')
    # thompson_sampling('Data Directory/reinforcement_learning.csv')

    # natural_language_processing('Data Directory/Restaurant_Reviews.tsv')

    # ann('Data Directory/ann.csv', 3, 2, 2)

    # NOTE: MUST UNZIP DATASET.ZIP PRIOR TO RUNNING conv_n_n
    conv_n_n('Data Directory/')

if __name__ == '__main__':
    main()