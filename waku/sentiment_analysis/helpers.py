import torch

# format the data, extracting the sentence as
# well as the sentiment of the entire sentence
def ReadTextFile(filepath) :
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
        
#     print("size of vocab in dataset:", len(dataVocab))
#     print("size of vocab in word2vec:", len(w2vVocab))
    # use the union between the data vocab and the word2vec vocab
    dataInW2V = dataVocab & w2vVocab
#     print("size of vocab union between data and word2vec", len(dataInW2V))

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

#     print("tokens in new embedding matrix", weights.shape[0])
    weights = weights.to(device)

    return  word2index, weights

# function to convert a list of words into the corresponding 
# indices using the supplied dictionary
def sentence2index(indexDict,sentence) :
    idx = list()
    for word in sentence :
        try :
            idx.append(indexDict[word])
        except:
            idx.append(indexDict["UNK"])
    return torch.tensor(idx, dtype=torch.long)

# function to combine data samples into a batch that the model can recieve
def pad_collate(batch):
    labels = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    # calculate the true lengths of each sequence
    phrase_lengths = torch.LongTensor([len(sentence) for sentence in text])
    # pad the sequences so that they are all of the same length to be processed
    # by the rnn as a batch
    text_pad = torch.nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value=0)
    return (text_pad, phrase_lengths), labels