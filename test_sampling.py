""" Tester code for student's implementation of the get_random_context()
    function from the dataset class. Using known random seeds, students
    should be able to reproduce a known list of words from the Stanford
    IMDB movie review dataset.
"""

import random
import numpy as np
from dataset import Dataset

if __name__ == '__main__':
    random.seed(1981)
    np.random.seed(1981)

    data = Dataset(fname='data/datasetSentences.txt')
    centers, contexts = [], []
    for _ in range(10):
        center_word, context = data.get_random_context()
        centers.append(center_word)
        contexts.append(context)

    assert centers == ['course', 'unfulfilled', 'pastel', 'purer', 'pessimists', 
                        'are', 'equation', '1999', 'raison', 'pathologically']

    assert contexts == [['abuse', 'veers', 'becomes', 'revenge'], 
                        ['perplexing', 'despite'], ['splash', 'prankish'], 
                        ['we'], ['half', 'totally'], 
                        ['weighed', 'supporting', 'goodly', 'knowing'], 
                        ['popular', 'box'], 
                        ['engaged', 'shameless', 'self-caricature', 'analyze', 
                            'analyze', 'promised', 'threatened', 'later'], 
                        ['kicking', "d'etre"], ['grave', 'purported', 'avenges', 'hatred']]

    print('all tests passed, your code is correct.')
