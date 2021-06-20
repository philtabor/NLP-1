""" Tester code for student's implementation of negative sampling. 
    Load the Stanford IMDB movie review dataset and generate 10
    negative samples using known random seeds.

    Please note, this code will fail after we have implemented the more
    efficient algorithm for generating negative indices.
"""

import random
import numpy as np
from dataset import Dataset

random.seed(1981)
np.random.seed(1981)
dataset = Dataset('data/datasetSentences.txt')

outside_word_idx = np.random.choice(dataset.vocabulary_size)

neg_samples = dataset.get_negative_samples(outside_word_idx, N=10)

assert neg_samples == [333, 13, 13528, 245, 18, 3975, 7321, 18777, 10, 15580]

print('The correct negative samples were generated.')
