import numpy as np
import random

def sigmoid(x):
    s = 1 / ( 1 + np.exp(-x))

    return s

def softmax(x):
    s = np.exp(x) / np.sum(np.exp(x))
    return s

class Word2Vec:
    def __init__(self, window_size, vocab_size, vec_size, word_mapping, dataset):
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.vec_size = vec_size
        self.word_2_ind = word_mapping
        self.dataset = dataset
        # Casting everything to np.float32 makes the code about 45% faster
        self.center_word_vectors = np.random.normal(size=(vocab_size,vec_size),
                                        loc=0.0, scale=0.01).astype(np.float32)
        self.outside_word_vectors = np.random.normal(size=(vocab_size,vec_size),
                                        loc=0.0, scale=0.01).astype(np.float32)

    def softmax_loss(self, center_word_vec, outside_word_idx):
        theta = np.dot(self.outside_word_vectors, center_word_vec)
        p = np.zeros((self.vocab_size,), dtype=np.float32)
        p[outside_word_idx] = 1.0
        p_hat = softmax(theta)
        loss = -np.sum(p*np.log(p_hat))
        error = p_hat - p
        
        grad_outside_vecs = np.outer(error, center_word_vec)
        grad_center_vec = np.dot(error, self.outside_word_vectors)

        return loss, grad_center_vec, grad_outside_vecs

    def neg_sampling_loss(self, center_word_vec, outside_word_idx, N=5):
        neg_sample_idx = self.dataset.get_negative_samples(outside_word_idx, N)
        grad_outside_vecs = np.zeros_like(self.outside_word_vectors,
                                          dtype=np.float32)
        w_c = center_word_vec
        w_o = self.outside_word_vectors[outside_word_idx]
        w_k = self.outside_word_vectors[neg_sample_idx]
        theta = np.dot(w_o.T, w_c)
        loss = -np.log(sigmoid(theta)) - \
                np.sum(np.log(sigmoid(-np.dot(w_c, w_k.T))))

        grad_center_vec = -(1-sigmoid(theta))*w_o
        grad_outside_vecs[outside_word_idx] = -(1-sigmoid(theta))*w_c
        for idx in neg_sample_idx:
            vector = self.outside_word_vectors[idx]
            beta = np.dot(-w_c, vector.T)
            grad_center_vec += np.dot((1-sigmoid(beta)), vector)
            grad_outside_vecs[idx] += (1-sigmoid(beta))*w_c
        return loss, grad_center_vec, grad_outside_vecs

    def skipgram(self,current_center_word, outside_words, 
                    loss_and_gradient_f=None):
        loss = 0.0
        grad_center_vecs = np.zeros(self.center_word_vectors.shape, 
                                    dtype=np.float32)
        grad_outside_vectors = np.zeros(self.outside_word_vectors.shape, 
                                    dtype=np.float32)

        w_c_idx = self.word_2_ind[current_center_word]
        w_o_idx = [self.word_2_ind[word] for word in outside_words]
        w_c = self.center_word_vectors[w_c_idx]

        for idx in w_o_idx:
            J, dJ_dwc, dJ_dwo = loss_and_gradient_f(center_word_vec=w_c,
                                                    outside_word_idx=idx)
            loss += J
            grad_center_vecs[w_c_idx] += dJ_dwc
            grad_outside_vectors += dJ_dwo
        return loss, grad_center_vecs, grad_outside_vectors

    def apply_gradients(self, grad_v, grad_u, learning_rate=0.1):
        self.outside_word_vectors -= learning_rate * grad_u
        self.center_word_vectors -= learning_rate * grad_v
