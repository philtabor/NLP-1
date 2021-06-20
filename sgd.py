import numpy as np

class SGDWrapper:
    def __init__(self, n_iterations=40000, batch_size=50, 
                 anneal_interval=20000, print_interval=1):
        self.n_iter = n_iterations
        self.batch_size = batch_size
        self.anneal_interval = anneal_interval
        self.print_interval = print_interval

    def sgd(self, dataset, word2vec, loss_fn, 
            learning_rate, lr_decay, window_size):
        for iteration in range(self.n_iter):
            loss = 0.0
            grad_v = np.zeros_like((word2vec.center_word_vectors))
            grad_u = np.zeros_like((word2vec.outside_word_vectors))
            exploss = None
            for mini_batch in range(self.batch_size):
                center_word, context = \
                   dataset.get_random_context(window=window_size)
                cost, d_v, d_u = word2vec.skipgram(center_word, context, 
                                    loss_and_gradient_f=loss_fn)
                loss += cost / self.batch_size

                grad_v += d_v / self.batch_size
                grad_u += d_u / self.batch_size
            word2vec.apply_gradients(grad_v, grad_u, learning_rate)
            if iteration % self.print_interval == 0 and iteration > 0:
                                
                if exploss is None:
                    exploss = loss
                else:
                    exploss = 0.95 * exploss + 0.05 * loss
                print('finish iteration ', iteration, 'total loss {:.1f}'.format(exploss))
            if iteration % self.anneal_interval == 0 and iteration > 0:
                learning_rate *= lr_decay
        return word2vec
