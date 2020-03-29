import torch
from torch.utils.data import Dataset

from waku.sentiment_analysis.helpers import sentence2index, ReadTextFile

class sentenceDataset(Dataset):
    """
    Custom dataset which processes the data usingthe defined functions,
    ready to be used in a dataloader
    """
    def __init__(self, ys, Xs, indexDict):
        """
        Arguments:
        ----------
        ys: `list` of `int`
            List of sentence labels
        Xs: `list` of `list` of `str`
            List of sentences, each a list of words
        indexDict: `dict`
            Dict mapping words to int, which indexes row of weight matrix 
        """
        self.y_train = ys
        # convert the sentences of words into indexes
        self.X_train = [sentence2index(indexDict, sentence) for sentence in Xs]
        
    def __len__(self):
        return len(self.X_train)
    
    def __getitem__(self, idx):
        return [self.y_train[idx], self.X_train[idx]]
    
class SST:
    """
    Stanford Sentiment Treebank data wrapper for experiment
    """
    def __init__(self, data_filepath):
        """
        Arguments:
        ----------
        data_filepath : `str`
            Filepath for raw text data
        """
        self.y_train, self.X_train = ReadTextFile(data_filepath+"train.txt")
        self.y_val, self.X_val = ReadTextFile(data_filepath+"dev.txt")
        self.y_test, self.X_test = ReadTextFile(data_filepath+"test.txt")

    def embed(self, embeddings_dict):
        """
        Create DataLoaders, embed raw sentences
        """
        self.TrainData = sentenceDataset(self.y_train, self.X_train, embeddings_dict)
        self.ValData = sentenceDataset(self.y_val, self.X_val, embeddings_dict)
        self.TestData = sentenceDataset(self.y_test, self.X_test, embeddings_dict)
    
    def reset(self):
        """
        Remove embedded sentences, ready for new experiment with new embeddings
        """
        self.TrainData = None
        self.ValData = None
        self.TestData = None    