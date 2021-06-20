""" Tester code for student's implementation of the stochastic gradient
    descent class. Use a dummy dataset to generate word vectors and 
    systematically apply gradients.
"""

import random
import numpy as np
from word2vec import Word2Vec
from sgd import SGDWrapper

class Dataset:
    def __init__(self):
        self.vocabulary = ['Q', 'W', 'X', 'Y', 'Z']
        self.vocabulary_size = len(self.vocabulary)
        self.word_map = {'Q':0, 'W':1, 'X':2, 'Y':3, 'Z':4}

    def get_random_context(self, window=4):
        context = [random.choice(self.vocabulary) for _ in range(window)]
        center_word = random.choice(self.vocabulary)
        return center_word, context

if __name__ == '__main__':
    random.seed(1981)
    np.random.seed(1981)
    dataset = Dataset()
    n_words = dataset.vocabulary_size
    n_dims = 4
    window_size = 3
    learning_rate = 0.3
    lr_decay = 0.5

    word2vec = Word2Vec(window_size=window_size, 
                        vocab_size=n_words, vec_size=n_dims,
                        word_mapping=dataset.word_map, dataset=dataset)

    sgd_wrapper = SGDWrapper(n_iterations=10, batch_size=50)
    word2vec = sgd_wrapper.sgd(dataset, word2vec, word2vec.softmax_loss, 
                               learning_rate, lr_decay, window_size)

    assert np.allclose(word2vec.center_word_vectors,
                        [[-0.00300049, -0.00677350, 0.015835040, -0.0093782],
                         [0.010229870,  0.02110826, -0.01064797,  0.0032986],
                         [-0.00681552,  0.01547351, 0.005442900,  0.00230276],
                         [0.007489810, -0.01501501, -0.00691206,  0.00631761],
                         [-0.00870971, -0.01209738, -0.00275560,  0.01015395]])

    assert np.allclose(word2vec.outside_word_vectors,
                        [[-0.00401956, -0.00111577, -0.00928379,  0.00605264],
                         [ 0.00644379,  0.00109810,  0.00961013, -0.00048445],
                         [-0.00430296,  0.01015337,  0.01049627,  0.00199262],
                         [-0.01787026,  0.00770430,  0.00269429,  0.01141347],
                         [ 0.00182249, -0.01437133,  0.00154702,  0.00722699]])
    print('Your word vectors matched expected values.')
