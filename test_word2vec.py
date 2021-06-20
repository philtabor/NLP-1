""" 
    File to test the student's code for the softmax loss and gradients, as well
    as the negative sampling loss and gradients.
"""
from word2vec import Word2Vec
import random
import numpy as np

class Dataset:
    def __init__(self):
        self.vocabulary = ['Q', 'W', 'X', 'Y', 'Z']
        self.word_map = {'Q':0, 'W':1, 'X':2, 'Y':3, 'Z':4}

    def get_random_context(self, window=4):
        context = [random.choice(self.vocabulary) for _ in range(window)]
        return context

    def get_negative_samples(self, outside_word_idx, N=5):
        neg_sample_word_indices = []
        for n in range(N):
            new_idx = outside_word_idx
            while new_idx == outside_word_idx:
                new_idx = self.word_map[random.choice(self.vocabulary)]
            neg_sample_word_indices.append(new_idx)
        return neg_sample_word_indices

random.seed(1981)
np.random.seed(1981)
data = Dataset()
word2vec = Word2Vec(window_size=3, vocab_size=5, word_mapping=data.word_map,
        dataset=data, vec_size=3)

def test_loss(loss_fn, algo):
    loss = 0
    d_center = np.zeros((5, 3))
    d_outside = np.zeros((5,3))

    center_word = 'Q'

    for _ in range(5):
        context = data.get_random_context()
        cost, grad_center, grad_outside = word2vec.skipgram(center_word, 
                                 context, loss_and_gradient_f=loss_fn)
        loss += cost
        d_center += grad_center
        d_outside += grad_outside

    if algo == 'softmax':
        assert loss == 32.18999207019806 

        assert np.allclose(d_center, [[ 0.03295639, -0.04433317, 0.06506403],
                                     [ 0., 0., 0.],
                                     [ 0., 0., 0.],
                                     [ 0., 0., 0.],
                                     [ 0., 0., 0. ]])

        assert np.allclose(d_outside, [[ 2.72225589e-06,  4.20096330e-06, -1.48280524e-05],
                                       [ 1.26745565e-02,  1.95621324e-02, -6.90336246e-02],
                                       [-1.50984852e-06, -2.32795719e-06,  8.22357833e-06],
                                       [-9.50546726e-03, -1.46709055e-02,  5.17727742e-02],
                                       [-3.17030580e-03, -4.89310629e-03,  1.72674861e-02]])
            
            
    elif algo == 'negative_sampling':
        assert loss == 83.18010543199095 

        assert np.allclose(d_center,  [[ 0.09446272, -0.15653473, 0.11456128],
                                      [0., 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., 0.]])

        assert np.allclose(d_outside, [[-0.0316767,  -0.04889038,  0.17253131],
                                       [-0.01267017, -0.01955536,  0.06900974],
                                       [-0.02059529, -0.03178713,  0.11217493],
                                       [-0.02851549, -0.04401131,  0.15531335],
                                       [-0.03326891, -0.05134782,  0.1812035 ]])
            
    print('All checks for', algo, 'complete. Your code passed.')


test_loss(word2vec.softmax_loss, 'softmax')
test_loss(word2vec.neg_sampling_loss, 'negative_sampling')
