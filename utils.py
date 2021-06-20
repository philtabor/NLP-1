import numpy as np
import matplotlib.pyplot as plt

def plot_words(ds, w_vecs, fname):
    """ This function is taken from the Stanford course
        CS224n, assignment 2. 
    """
    words = ["entertaining", "fun", "difficult", 
                      "original", "quality", "simple", 
                      "boring", "scary",  
                      "action", "drama", "derivative", "exciting"]

    indices = [ds.word_map[word] \
                for word in words]
    vectors = w_vecs[indices, :] 

    norm_vectors = vectors - np.mean(vectors, axis=0)
    cov = 1.0 / len(indices) * norm_vectors.T.dot(norm_vectors)
    U,S,V = np.linalg.svd(cov)
    coord = norm_vectors.dot(U[:,0:2])
    
    for i in range(len(words)):
        plt.text(coord[i,0], coord[i,1], words[i],
                bbox=dict(facecolor='green', alpha=0.1))
    plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
    plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))
    plt.savefig(fname)
    # if we don't close the plot, it will retain the plot each time it's called
    plt.close()

def get_analogy(w_a, w_b, w_c, ds, vectors):
    w_a_idx = ds.word_mappings[w_a]
    w_b_idx = ds.word_mappings[w_b]
    w_c_idx = ds.word_mappings[w_c]

    analogy_indices = [w_a_idx, w_b_idx, w_c_idx]

    v_a = vectors[w_a_idx]
    v_b = vectors[w_b_idx]
    v_c = vectors[w_c_idx]

    projection = np.transpose(v_b - v_a + v_c) / np.abs(v_b - v_a + v_c)

    indices = [ds.word_mappings[word] for word in ds.word_mappings]
    for idx in analogy_indices:
        indices.remove(idx)

    words = np.dot(vectors[indices], projection)
    w_d_idx = np.argmax(words, axis=0)

    return ds.vocabulary[w_d_idx]
