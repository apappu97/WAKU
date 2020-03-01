import numpy as np
import random


class DataPipeline:
    def __init__(self, data, vocabs, word_freq, table_size, use_noise_neg=True):
        """
        Arguments:
        ----------
        data : `list` of `int`, [word_index]
            The corpus where each word is mapped to an integer.
        vocabs : `list` of `int`
            List of all unique integers used to map words to integers.
        word_freq : `dict`
            Dictionary of frequency of each word in corpus. 
        table_size : `int`
            Size of the table for generating negative samples.            
        use_noise_neg : `Boolean`
            Generate noise distribution for negative sampling.
        """
        self.data = data
        # Whether to obtain negative samples from a noise distirbution or 
        # sample the negative samples uniformly
        if use_noise_neg:
            self.unigram_table = noise(vocabs, word_freq, table_size)
        else:
            self.unigram_table = vocabs

    def get_neg_data(self, batch_size, num_neg_samples, target_inputs):
        """
        Sample the negative data.

        Parameters:
        -----------
        batch_size : `int`
            Number of word pairs in each batch.  
        num_neg_samples : `int`
            Number of negative samples for each center word.
        target_input : `list` of `int`
            List of center words.

        Returns:
        --------
        neg_sampled_context_words : `numpy.ndarray`
            (batch_size, num_neg_samples) Matrix of negative samples for center 
            words in batch.
        """
        neg = np.zeros((num_neg_samples))
        for i in range(batch_size):
            # Randomly sample words from unigram table for negative samples
            delta = random.sample(self.unigram_table, num_neg_samples)
            # Make sure center word is not in negative samples.
            while target_inputs[i] in delta:
                delta = random.sample(self.unigram_table, num_neg_samples)
            neg = np.vstack([neg, delta])
        return neg[1: batch_size + 1]

    def generate_batch(self, batch_size, window_size, data_index):
        """
        Generate batches of data

        Parameters:
        -----------
        batch_size : `int`
            Number of word pairs in each batch. 
        window_size : `int`
            Max skip length between words. 
        data_index : `int`
            The index of the data the batch should start from.

        Returns:
        --------
        batch_x : `numpy.ndarray`
            (batch_size) Batch of the center words.
        batch_y : `numpy.ndarray`
            (batch_size) Batch of the corresponding context words in the window_size.
        data_index : `int`
            The index of the data the generator has got to.           
        """
        n_words = len(self.data)
        center_words, context_words = list(), list()
        while data_index != n_words:
            # Place center word in list
            x = [self.data[data_index]]
            # Get context words in window size [window_size, center_word, window_size]
            y = self.data[max(0, data_index-window_size):data_index] + self.data[data_index+1:min(data_index+1+window_size, n_words)]
            # Add to batch lists
            center_words.extend(x * len(y))
            context_words.extend(y)
            # Move onto next center word
            data_index += 1
            # If batch is full yield
            if len(center_words) > batch_size:
                batch_y = context_words[:batch_size]
                batch_x = center_words[:batch_size]
                center_words = center_words[batch_size:]
                context_words = context_words[batch_size:]
                yield np.array(batch_x), np.array(batch_y), data_index
            # If on last word and batches are not empty yield
            elif data_index == n_words and len(center_words)!=0:
                yield np.array(center_words), np.array(context_words), data_index