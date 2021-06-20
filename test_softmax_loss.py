""" Tester code for student's implementation of the softmax loss and
    gradient function. Using a known random seed, they should be able to 
    reproduce the given loss and gradients. 
"""

from word2vec import Word2Vec
import numpy as np
import random

class Dataset:
    def __init__(self):
        self.vocabulary = ['Q', 'W', 'X', 'Y', 'Z']
        self.word_map = {'Q':0, 'W':1, 'X':2, 'Y':3, 'Z':4}

random.seed(1981)
np.random.seed(1981)
data = Dataset()

word2vec = Word2Vec(window_size=3, vocab_size=5, word_mapping=data.word_map,
        dataset=data, vec_size=3)

center_vec = np.random.randn(3)

loss, grad_center, grad_outside = word2vec.softmax_loss(center_vec, 1)

assert loss == 1.599380520422519

assert np.allclose(grad_center, [ 0.00475941, -0.0125002, 0.00796234])

assert np.allclose(grad_outside, [[ 0.23835036,  0.02982749, -0.35162963],
                                  [-0.92508339, -0.1157662,   1.36474193],
                                  [ 0.2289308,  0.02864871, -0.3377333 ],
                                  [ 0.23029447,  0.02881937, -0.33974506],
                                  [ 0.22750775,  0.02847063, -0.33563393]])
    
print('All tests passed; it would appear your code is correct.')
