import numpy as np
import collections
import pickle
import random
import time

class DataProcessor:
    def __init__(self, input_filename, thresh_subsample=1e-4, min_count=0,
                 window_size=5, num_neg_samples=5, table_size=1e8, use_noise_neg=True):
        """
        Arguments:
        ----------
        input_filename : `str`
            The input filename for the data. 
        thresh_subsample : `float`
            Subsampling threshold.
        min_count : `int`
            Frequency of words to remove.              
        window_size : `int`
            Max skip length between words.            
        num_neg_samples : `int`
            Number of negative samples for each center word.
        table_size : `int`
            Size of the table for generating negative samples.                                   
        use_noise_neg : `Boolean`
            Generate noise distribution for negative sampling.
        """
        # Get data, vocabulary size, word frequency, word2index and index2word mappings
        # Removes infrequent words and sub-sampling of frequent words
        self.data, self.word_freq, self.word2index, self.index2word = self.build_dataset(self.read_data(input_filename), min_count, thresh_subsample)        
        # List of all unique indices used to map words to indices
        self.vocabs = list(set(self.data))    
        # Whether to obtain negative samples from a noise distirbution or 
        # sample the negative samples uniformly
        if use_noise_neg:
            self.unigram_table = self.noise(self.vocabs, self.word_freq, table_size)
        else:
            self.unigram_table = self.vocabs
        # Preprocess data to get center words, context words and negative samples
        self.center_words, self.context_words, self.neg_samples = self.preprocess_data(self.data, window_size, num_neg_samples)

    def build_dataset(self, corpus, min_count, thresh_subsample):
        """
        -Maps corpus of words to indices.
        -Subsamples of frequent words.
        -Remove infrequent words.

        Parameters:
        -----------
        corpus : `list` of `str`
            The input data.
        min_count : `int`
            Frequency of words to remove. 
        thresh_subsample : `float`
            Subsampling threshold.                  
        
        Returns:
        --------
        data : `list` of `int`, [word_index]
            Corpus (strings) converted to corpus (indices).
        word_freq  : `dict`
            Dictionary of frequency of each word in corpus.
        word2index : `dict`, {word_str: word_index}
            Mapping from word to index.
        index2word : `dict`, {word_index: word_str}
            Mapping from index to word.
        """
        print("Length of corpus before removing infrequent words and subsampling: {:,}".format(len(corpus)))  
        print("Vocabulary size: {:,}".format(len(collections.Counter(corpus)))) 
        # Get frequency of each word in corpus
        word_freq = dict(collections.Counter(corpus))
        # Remove words if frequency is less than min_count
        corpus = [word for word in corpus if word_freq[word] > min_count]
        # Get number of unique characters in corpus - vocabulary size
        n_words = len(collections.Counter(corpus)) 
        print("Length of corpus after removing infrequent words: {:,}".format(len(corpus)))
        print("Vocabulary size: {:,}".format(n_words)) 
        # Get frequency of each word in corpus after removing infrequent words   
        word_freq = [('UNK', -1)]
        word_freq.extend(collections.Counter(corpus).most_common(n_words - 1))
        word_freq = dict(word_freq)    
        # Create mapping from word to index
        word2index = {key:i for i, key in enumerate(word_freq.keys())}  
        # Get data and count for UNK 
        data = list()
        unk_count = 0
        for word in corpus:
            if word in word2index:
                index = word2index[word]
            else:
                # Replacing word not in corpus with UNK index of 0
                index = 0
                unk_count += 1
            data.append(index)
        # Update count for UNK
        word_freq['UNK'] = unk_count
        # Get reverse mapping, from index to word
        index2word = dict(zip(word2index.values(), word2index.keys())) 
        # Sub-sampling of frequent words  
        data, word_freq = self.subsample(data, word_freq, thresh_subsample)
        print("Length of corpus after subsampling: {:,}".format(len(data)))
        return data, word_freq, word2index, index2word

    def preprocess_data(self, data, window_size, num_neg_samples):
        """
        Preprocess data:
        1. For each word in data (center word) get corresponding context words
        2. For each word in data (center word) get negatives samples.

        Parameters:
        -----------
        data : `list` of `int`, [word_index]
            The corpus where each word is mapped to an integer.
        window_size : `int`
            Max skip length between words.            
        num_neg_samples : `int`
            Number of negative samples for each center word.

        Returns:
        --------
        center_words : `numpy.ndarray`
            List of center words from the data.
        context_words : `numpy.ndarray`
            List of corresponding context words.
        neg_samples : `numpy.ndarray`
            (len(center_words), num_neg_samples) Matrix of negative samples 
            for all center words.
        """
        start_time = time.time()
        n_words = len(data)
        neg_samples = [[0]*num_neg_samples]
        center_words, context_words = list(), list()
        # Get center words and corresponding context words
        for i, word in enumerate(data):

            # Place center word in list
            x = [word]
            # Get context words in window size [window_size, center_word, window_size]
            y = data[max(0, i-window_size):i] + data[i+1:min(i+1+window_size, n_words)]            
            # Add to lists
            center_words.extend(x*len(y))
            context_words.extend(y)
            # Combine center word and context words
            x.extend(y)
            # Get negative samples for center words
            for j in range(len(y)):
                # Randomly sample words from unigram table for negative samples
                delta = random.sample(self.unigram_table, num_neg_samples)
                # Make sure center word or context words are not in negative samples.
                while any(word in delta for word in x):
                    delta = random.sample(self.unigram_table, num_neg_samples)
                neg_samples.append(delta)

        print("Time Taken for preprocessing of data: {:,.4f} seconds...".format(time.time()-start_time),
              "Number of training exmaples: {:,}".format(len(center_words)))
        return np.array(center_words), np.array(context_words), np.array(neg_samples[1: len(center_words) + 1])

    # Helper functions
    #-----------------
    def save_data(self, filename):
        """
        Save center words and correspodning context_words 
        and negatively sampled words to .npz (compressed) format

        Parameters:
        -----------
        filename: `str`
            The filename to dump the data.
        """
        np.savez_compressed(filename, 
                            a=self.center_words, 
                            b=self.context_words,
                            c=self.neg_samples)       

    def save_word2index(self, filename):
        """
        Pickle word2index dictionary.

        Parameters:
        -----------
        filename: `str`
            The filename to dump the word2index dictionary.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.word2index, f)    

    def save_index2word(self, filename):
        """
        Pickle index2word dictionary.

        Parameters:
        -----------
        filename: `str`
            The filename to dump the index2word dictionary.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.index2word, f)                   

    def save_word_freq(self, filename):
        """
        Pickle word_freq dictionary.

        Parameters:
        -----------
        filename: `str`
            The filename to dump the word frequency dictionary.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.word_freq, f) 

    # Static Methods
    #---------------
    @staticmethod
    def read_data(filename):
        """
        Converts data in filename to list of words.

        Parameters:
        -----------
        filename : `str`
            Filename for input data.

        Returns:
        --------
        data : `list` of `str`
            Data converted to list.
        """
        with open(filename, 'r', encoding='utf-8') as f:
            data = f.read().split()
            # Check if all word strings are lowercase
            if any(word for word in data if word.isupper()):
                print("Upper case letter detected in corpus")
            else:
                print("All words in corpus are lower case")
        return data   

    @staticmethod
    def subsample(data, word_freq, thresh_subsample):
        """
        Subsampling of frequent words. Each word w_i in the training set is kept with probability 
        P(w_i) = t/f(w_i)*[1 + sqrt{f(w_i)/t}]
        where f(w_i) is the normalised frequency of word w_i and t is a chosen threshold.

        Parameters:
        -----------
        data : `list` of `int`
            The corpus where words are mapped to indices.
        word_freq : `dict`
            Dictionary of frequency of each word in corpus.
        thresh_subsample : `float`
            Subsampling threshold.        

        Returns:
        --------
        subsampled_data : `list` of `int`
            The data after subsampling of frequent words.
        subsampled_word_freq : `dict`
            Dictionary of frequency of each word in corpus after subsampling.
        """
        freq = list(word_freq.values())
        freq = np.array(freq) / sum(freq)
        # Get probability of discarding words
        P = thresh_subsample/freq * (1 + np.sqrt(freq/thresh_subsample))
        subsampled_data = list()
        # Subsample frequent words
        subsampled_data = [word for word in data if random.random() < P[word]]
        # Create dictionary of subsampled word frequency where keys are the indices
        subsampled_word_freq = collections.Counter(subsampled_data)
        # Create dictionary of subsampled word frequency where keys are the word strings
        subsampled_word_freq = {key:subsampled_word_freq[i] for i, key in enumerate(word_freq.keys())}
        # Raise exception if words obtain zero frequency
        if 0 in subsampled_word_freq.values():
            raise Exception("Zero frequency found")
        return subsampled_data, subsampled_word_freq   

    @staticmethod
    def noise(vocabs, word_freq, table_size):
        """
        Generate noise distribution to sample the negative samples from.
        P(w_i) = f(w_i)^{3/4} / \sum f(w_i)^{3/4}
        where f(w_i) is the frequency of words w_i in the corpus.

        For practically we fix the size of the unigram table and fill this table 
        with the index of each word in the vocabulary multiple times. The number of 
        times a word’s index appears in the table is given by P(w_i) * table_size.

        Parameters:
        -----------
        vocabs : `list` of `int`
            List of all unique integers used to map words to integers.
        word_freq : `dict`
            Dictionary of frequency of each word in corpus. 
        table_size : `int`
            Size of the table for generating negative samples.         

        Returns:
        --------
        unigram_table : `list` of `int`
            Generate distribution for sampling negative words.
        """
        unigram_table = list()
        raised_counts = np.fromiter(word_freq.values(), dtype=int)**0.75
        total_raised_counts = sum(raised_counts)
        for index in vocabs:
            unigram_table.extend([index] * int(table_size * (raised_counts[index] / total_raised_counts)))
        return unigram_table                    