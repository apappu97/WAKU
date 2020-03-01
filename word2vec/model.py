import torch
from torch import nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    """
    Skip gram word2vec model
    """
    def __init__(self, vocab_size, emb_dimension):
        """
        Arguments:
        ----------
        vocab_size : `int`
            Vacabulary size -  size of the dictionary.
        emb_dimension : `int`
            Embedding dimension - the size of each embedding vector (50-500).

        input_embedding: Embedding for center word (input word).
        output_embedding: Embedding for neighbour/context words (output word).
        """
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.input_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)
        self.output_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)

        # Initialise embedding weights like word2vec
        # The input_embedding is a uniform distribution in [-0.5/em_size, 0.5/emb_size]
        # The elements of output_embedding are zeroes
        initrange = 0.5 / self.emb_dimension
        self.input_embeddings.weight.data.uniform_(-initrange, initrange)
        self.output_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, center_words, context_words, neg_context_words):
        """
        Forward pass through the network.

        Parameters:
        -----------
        center_words: `torch.Tensor`, [batch_size] 
            List of center word indices for positive word pairs.
        context_words: `torch.Tensor`, [batch_size]
            List of neighbour word indices for positive word pairs.
        neg_context_words: `torch.Tensor`, [batch_size, neg_sampling_count]
            List of neighbour word indices for negative word pairs.

        Returns:
        --------
        loss : `torch.Tensor`
            The loss of the forward pass using negative sampling.
        """
        # Convert word indices to embeddings [batch_size, emb_dimension], 
        # [batch_size, neg_sampling_count, emb_dimension]
        emb_center = self.input_embeddings(center_words)
        emb_context = self.output_embeddings(context_words)
        emb_neg_context = self.output_embeddings(neg_context_words)

        # Calculating first term of loss function
        pos_score = F.logsigmoid(torch.sum(emb_center * emb_context, dim=1)).squeeze()
        # Calculating second term of loss function
        neg_score = torch.bmm(emb_neg_context, emb_center.unsqueeze(2)).squeeze(2)
        neg_score = F.logsigmoid(-torch.sum(neg_score, dim=1)).squeeze()
        # Calculate final loss
        return -(pos_score + neg_score).mean()

    def predict(self, inputs):
        """
        Map input indices to embeddings.

        Parameters:
        -----------
        inputs : `torch.Tensor`, [batch_size]
            List of word indices to map to embedding.

        Returns:
        --------
        prediction : `torch.Tensor`, [batch_size, emb_dimension]
            Embedded representation of input.
        """
        return self.input_embeddings(inputs)         