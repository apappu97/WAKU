class DataPipeline:
    def __init__(self, center_words, context_words, neg_samples):
        """
        Class for batching the data.

        Arguments:
        ----------
        center_words : `numpy.ndarray`
            List of center words from the data.
        context_words : `numpy.ndarray`
            (len(center_words)) List of corresponding context words.
        neg_samples : `numpy.ndarray`
            (len(center_words), num_neg_samples) Matrix of negative samples 
            for all center words.    
        """       
        self.center_words = center_words
        self.context_words = context_words
        self.neg_samples = neg_samples
        # The index of the data the batch should start from.   
        self.data_index = 0

    def generate_batch(self, batch_size):
        """
        Generate batches of data

        Parameters:
        -----------
        batch_size : `int`
            Number of word pairs in each batch. 
        
        Returns:
        --------
        center_words: `numpy.ndarray`
            (batch_size) Batch of the center words.
        context_words : `numpy.ndarray`
            (batch_size) Batch of the corresponding context words in the window_size.
        neg_samples : `numpy.ndarray`
            (batch_size, num_neg_samples) Batch of the corresponding negative samples.           
        """
        n_words = len(self.center_words)
        while self.data_index <= n_words:
            self.data_index += batch_size
            yield self.center_words[self.data_index-batch_size:self.data_index], self.context_words[self.data_index-batch_size:self.data_index], self.neg_samples[self.data_index-batch_size:self.data_index, :]        