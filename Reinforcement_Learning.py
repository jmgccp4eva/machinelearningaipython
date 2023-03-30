import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

def upper_confidence_bound(file):
    dataset = pd.read_csv(file)
    d = 10
    N = 10000
    ads_selected = []
    numbers_of_selections = [0] * d
    sums_of_rewards = [0] * d
    total_reward = 0
    for n in range(0, N):
        ad = 0
        max_upper_bound = 0
        for i in range(0, d):
            if (numbers_of_selections[i] > 0):
                upper_bound = (sums_of_rewards[i] / numbers_of_selections[i]) + \
                             (math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i]))
            else:
                upper_bound = 1e400
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i
        ads_selected.append(ad)
        numbers_of_selections[ad] += 1
        reward = dataset.values[n, ad]
        sums_of_rewards[ad] += reward
        total_reward += reward

    # Visualizing results
    plt.hist(ads_selected)
    plt.title('Histogram of Ad Selections')
    plt.xlabel('Ads')
    plt.ylabel('Number of times each ad selected')
    plt.savefig('Images/UCB.png')
    plt.show()

def thompson_sampling(file):
    dataset = pd.read_csv(file)

    # Implementing Thompson Sampling
    d = 10
    N = 10000
    ads_selected = []
    number_of_rewards1 = [0] * d
    number_of_rewards0 = [0] * d
    total_rewards = 0
    for n in range(0, N):
        ad = 0
        max_random = 0
        for i in range(0, d):
            random_beta = random.betavariate(number_of_rewards1[i] + 1, number_of_rewards0[i] + 1)
            if random_beta > max_random:
                max_random = random_beta
                ad = i
        ads_selected.append(ad)
        reward = dataset.values[n, ad]
        if reward == 1:
            number_of_rewards1[ad] += 1
        else:
            number_of_rewards0[ad] += 1
        total_rewards += reward

    # Visualize Histogram of results
    plt.hist(ads_selected)
    plt.title('Histogram of Ad Selections')
    plt.xlabel('Ads')
    plt.ylabel('Number of times each ad selected')
    plt.savefig('Images/Thompson_Sampling.png')
    plt.show()