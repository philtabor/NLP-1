""" simple script to demonstrate the difference
    between the frequency and frequency**0.75, for
    the negative sampling.
"""
import numpy as np
vocabulary = ['A', 'B', 'C', 'D']

corpus = [['A', 'C', 'C', 'C', 'D', 'D'],
          ['D', 'C', 'C', 'C', 'C', 'C'],
          ['B', 'C', 'A', 'D', 'C', 'C'],
          ['A', 'B', 'D', 'C', 'C', 'C']]

word_counts = {}
for word in vocabulary:
    word_counts[word] = 0

for sentence in corpus:
    for word in sentence:
        word_counts[word] += 1

sample_freq = np.zeros((len(vocabulary),))
sample_freq_linear = np.zeros((len(vocabulary),))

for idx, word in enumerate(vocabulary):
    frequency = word_counts[word] 
    sample_freq_linear[idx] = frequency
    frequency = frequency ** 0.75
    sample_freq[idx] = frequency

sample_freq /= np.sum(sample_freq)
sample_freq_linear /= np.sum(sample_freq_linear)

for idx, word in enumerate(vocabulary):
    print('word: %s \t count: %s \t freq: %.2f \t freq**0.75: %.2f' %
            (word, word_counts[word], sample_freq_linear[idx], sample_freq[idx]))
