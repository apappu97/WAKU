import torch

def ReadTextFile(filepath) :
    """
    Extract raw sentences and sentiment labels from data

    Parameters:
    ----------- 
    filepath : `str`
        The path of data file

    Returns:
    --------
    y : `list` of `int`
        List of sentiment labels
    X : `list` of `list` of `str
        List of sentences, each itself a list of words

    """
    y = list()
    X = list()
    with open(filepath) as r :
        for line in r.read().split('\n') :
            #set_trace()
            if len(line)==0 :
                pass
            else :
                y.append(int(line[1]))
                X.append([word[:-1].replace(')','') for word in line.split() if word[-1]==')'])
    return y, X

def reduce_preprocess_embedding(x_train, x_val, x_test, embedding_dict, embedding_weights):
    """
    Removes vocabulary and corresponding embeddings of unused words which are reintroduced later at test time

    Parameters:
    ----------- 
    x_train : `list`
        List of train sentences
    x_val : `list`
        List of validation sentences
    x_test : `list`
        List of test sentences
    embeddings_dict : `dict`
        Dict mapping words to int, which indexes row of embeddings matrix  
    embeddings_weights : `numpy.ndarray``
        (len(vocab), 300) Matrix of word embeddings  

    Returns:
    --------
    word2index : `dict`
        Reduced embedding dictionary mapping words to int
    weights : `numpy.ndarray``
        (len(smaller_vocab), 300) Matrix of word embeddings  
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_dim = embedding_weights.shape[1]
    w2vVocab = set(embedding_dict.keys())
    dataVocab = set()
    word2index = dict()

    for i in range(len(x_train)):
        dataVocab |= set(x_train[i])
    for i in range(len(x_val)):
        dataVocab |= set(x_val[i])
    for i in range(len(x_test)):
        dataVocab |= set(x_test[i])       
        
    # use the union between the data vocab and the word2vec vocab
    dataInW2V = dataVocab & w2vVocab

    # for every word appearing in both datasets, copy the embedding into a new matrix
    weights = torch.empty((len(dataInW2V)+2, n_dim))
    for i, word in enumerate(dataInW2V):
        word2index[word] = i+2
        weights[i+2, :] = torch.from_numpy(embedding_weights[embedding_dict[word]])

    # add a pad token
    word2index["PAD"] = 0
    weights[0, :] = torch.zeros(1,n_dim)
    # add an unknown word token
    word2index["UNK"] = 1
    weights[1, :] = torch.from_numpy(embedding_weights[embedding_dict["UNK"]])

    weights = weights.to(device)

    return  word2index, weights

def sentence2index(indexDict,sentence) :
    """
    Convert a list of words into the corresponding indices using the supplied dictionary

    Parameters:
    -----------
    indexDict : `dict`
        Dict mapping words to int, which indices row of embeddings matrix
    sentence : `list` of `str`
        List of words

    Returns:
    --------
    torch.tensor(idx, dtype=torch.long) : `torch.Tensor`
        Tensor of mapped words to indices
    """
    idx = list()
    for word in sentence :
        try :
            idx.append(indexDict[word])
        except:
            idx.append(indexDict["UNK"])
    return torch.tensor(idx, dtype=torch.long)

def pad_collate(batch):
    """
    Combines data samples into a batch that the model can receive

    Parameters:
    -----------
    batch : `torch.Tensor`
        DataLoader item with embedded sentences and labels

    Returns:
    --------
    (text_pad, phrase_lengths) : `tuple` of (`torch.Tensor`, `int`)
        Padded sequence of embedded words, length of sentence

    labels : `int`
        Class label of sentence 
    """
    labels = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    # calculate the true lengths of each sequence
    phrase_lengths = torch.LongTensor([len(sentence) for sentence in text])
    # pad the sequences so that they are all of the same length to be processed
    # by the rnn as a batch
    text_pad = torch.nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value=0)
    return (text_pad, phrase_lengths), labels