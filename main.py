import random, time
import numpy as np
from word2vec import Word2Vec
from dataset import Dataset
from sgd import SGDWrapper
import matplotlib.pyplot as plt
from utils import plot_words, get_analogy

if __name__ == '__main__':
    random.seed(1981)
    dataset = Dataset('data/datasetSentences.txt')
    n_words = dataset.vocabulary_size
    n_dims = 25
    window_size = 5
    batch_size = 50
    n_iter = 20000
    learning_rate = 0.3
    lr_decay = 0.5
    ANNEAL_EVERY = 10000
    PRINT_INTERVAL = 100

    np.random.seed(1981)

    word2vec = Word2Vec(window_size=window_size, 
                        vocab_size=n_words, vec_size=n_dims,
                        word_mapping=dataset.word_map, dataset=dataset)
    print('... beginning stochastic gradient descent ...')
    sgd_wrapper = SGDWrapper(n_iterations=n_iter, batch_size=50, print_interval=PRINT_INTERVAL,
                            anneal_interval=ANNEAL_EVERY)
    start = time.time()
    word2vec = sgd_wrapper.sgd(dataset, word2vec, word2vec.neg_sampling_loss, 
                               learning_rate, lr_decay, window_size)
    end = time.time()
    delta = end - start
    print(n_iter, 'iterations took', delta, 'seconds to complete')
    figure_file = 'plots/25dims/word_vectors_final_test_center_word_vectors_randn_init_20000iters.png'

    #word_vectors = np.mean([word2vec.center_word_vectors, 
    #                        word2vec.outside_word_vectors], axis=0)

    plot_words(dataset, word2vec.center_word_vectors, figure_file) 
