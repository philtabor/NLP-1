import numpy as np
#import os
import random

class Dataset:
    def __init__(self, fname=None, source='stanford'):
        self.fname = fname
        self.source = source
        self.dataset = self.read_file_to_dataset()
        self.vocabulary, self.vocabulary_size = self.get_vocabulary()
        self.word_map = self.get_mappings()
        self.word_counts = self.get_word_counts()
        self.augmented_data = self.augment_data()

        self.size = len(self.dataset)
        self.weighted_samples = self.weight_words()

    def __len__(self):
        return self.size

    def read_file_to_dataset(self):
        print('... converting file to dataset ...')
        sentences = []
        reviews = open(self.fname).readlines(  )
        n_reviews = len(reviews)
        first = self.source == 'stanford' 
        offset = int(first)

        for line in range(n_reviews):
            if first:
                first = False
                continue
            sentence = [w.lower() for w in reviews[line].split()[offset:]]
            sentences.append(sentence)
        print('... compiled list of', len(sentences), 'sentences')
        return sentences

    def get_vocabulary(self):
        print('... tabulating vocabulary ...')
        vocabulary = []
        for sentence in self.dataset:
            for word in sentence:
                if word not in vocabulary:
                    vocabulary.append(word)

        #vocabulary = sorted(vocabulary)
        vocab_size = len(vocabulary)
        print('... found %d distinct words ...' % vocab_size)
        return vocabulary, vocab_size

    def get_mappings(self):
        print('... mapping words to integer indices ...')
        mapping = {}
        idx = 0
        for word in self.vocabulary:
            if word not in mapping:
                mapping[word] = idx
                idx += 1
        return mapping

    def get_word_counts(self):
        print('... getting word counts ...')
        unique_words = []
        word_counts = {}
        cnt = 0
        for word in self.vocabulary:
            word_counts[word] = 0

        for sentence in self.dataset:
            for word in sentence:
                word_counts[word] += 1

        return word_counts

    def augment_data(self, threshold=1e-5, N=30, min_length=3):
        print('... augmenting dataset ...')
        n_words = 0
        for word in self.word_counts:
            n_words += self.word_counts[word]

        reject_prob = np.zeros((self.vocabulary_size,), dtype=np.float32)
        for i in range(self.vocabulary_size):
            word = self.vocabulary[i]
            frequency = self.word_counts[word] / n_words
            reject_prob[i] = max(0, 1-np.sqrt(threshold/frequency))
                    
        all_sentences = []
        for sentence in self.dataset*N:
            new_sentence = []
            for word in sentence:
                if reject_prob[self.word_map[word]] == 0 or \
                        random.random() >= reject_prob[self.word_map[word]]:
                            new_sentence.append(word)
            if len(new_sentence) >= min_length:
                all_sentences.append(new_sentence)
       
        return all_sentences

    def get_random_context(self, window=5):
        sentence_number = random.randint(0, len(self.augmented_data)-1)
        sentence = self.augmented_data[sentence_number]
       
        # if our context length is too long, pick another
        random_context_length = random.randint(1, window)
        while len(sentence) < 2 * random_context_length + 1:
            #print(len(sentence), random_context_length)
            #input()
            random_context_length = random.randint(1, window)

        # alternative solution - discard the sentence and start over
        #if len(sentence) < 2 * random_context_length + 1:
        #    return self.get_random_context(window)
        
        center_word_idx = random.randint(random_context_length, 
                                        len(sentence)-random_context_length)
        low = center_word_idx - random_context_length
        # the upper limit is exclusive, so we have to add 1.
        high = center_word_idx + random_context_length + 1
        context = sentence[low:high]
        center_word = sentence[center_word_idx]
        context.remove(center_word)
        
        return center_word, context

    def get_negative_samples(self, outside_word_idx, N):
        neg_sample_word_indices = []
        for n in range(N):
            new_idx = outside_word_idx
            while new_idx == outside_word_idx:
                new_idx = self.sample_word_idx()
            neg_sample_word_indices.append(new_idx)
        return neg_sample_word_indices

    def weight_words(self):
        sample_freq = np.zeros((self.vocabulary_size,), dtype=np.float32)
        for idx, word in enumerate(self.vocabulary):
            frequency = self.word_counts[word]
            frequency = frequency ** 0.75
            sample_freq[idx] = frequency
        sample_freq /= np.sum(sample_freq)



        table_size = 1000000
        word_table = np.zeros(table_size, dtype=np.int32)
        samples = sample_freq * table_size
        w_table_idx = 0
        """
        for sample in samples:
            for _ in range(int(sample)):
                word_table[w_table_idx] = int(sample)
                w_table_idx += 1
        """
        for word_idx, sample in enumerate(samples):
            for _ in range(int(sample)):
                word_table[w_table_idx] = word_idx
                w_table_idx += 1
        return word_table
       
        #return sample_freq

    def sample_word_idx(self):
        #word = np.random.choice(self.vocabulary, p=self.weighted_samples)
        #idx = self.word_map[word]
        idx = np.random.choice(self.weighted_samples)
        return idx
