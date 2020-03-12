import requests
import gzip
import zipfile
from os import path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import copy
import re
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used: ", device)

# read pickle file into dictionary
def load_pickle(filepath):
    # file path must end with .pickle
    pickle_in = open(filepath,"rb")
    emb_dict = pickle.load(pickle_in)
    return emb_dict


# helper function to download the data and pretrained embeddings
def downloadFile(url,filepath) :
    if not path.exists(filepath) :
        with requests.get(url) as r :
            open(filepath, 'wb').write(r.content)
    if filepath[-3:]=='.gz' :
        if not path.exists(filepath[:-3]) :
            with gzip.open(filepath) as gz :
                open(filepath[:-3], 'wb').write(gz.read())
    if filepath[-4:]=='.zip' :
        if not path.exists(filepath[:-4]) :
            with zipfile.ZipFile(filepath,'r') as zp :
                zp.extractall()

# format the data, extracting the sentence as
# well as the sentiment of the entire sentence
def ReadTextFile(filepath) :
    y = []
    X = []
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
        
    print("size of vocab in dataset:", len(dataVocab))
    print("size of vocab in word2vec:", len(w2vVocab))
    # use the union between the data vocab and the word2vec vocab
    dataInW2V = dataVocab & w2vVocab
    print("size of vocab union between data and word2vec", len(dataInW2V))

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

    print("tokens in new embedding matrix", weights.shape[0])
    weights = weights.to(device)

    return  word2index, weights

# function to convert a list of words into the corresponding 
# indices using the supplied dictionary
def sentence2index(indexDict,sentence) :
    idx = []
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

# custom dataset which processes the data using the defined functions,
# ready to be used in a dataloader
class sentenceDataset(Dataset):
    def __init__(self, ys, Xs, indexDict):
        self.y_train = ys
        # convert the sentences of words into indexes
        self.X_train = [sentence2index(indexDict, sentence) for sentence in Xs]
        
    def __len__(self):
        return len(self.X_train)
    
    def __getitem__(self, idx):
        return [self.y_train[idx], self.X_train[idx]]

"""## Creating + training the model"""
class LSTMLearner(nn.Module):
    def __init__(self, embeddingDict, embeddingWeights, hidden_size=300, rnn_layers=1,
                 mlp_layer_widths=100):
        super(LSTMLearner, self).__init__()
        #use the pretrained embeddings
        self.n_dim = embeddingWeights.shape[1]
        self.embedding = nn.Embedding.from_pretrained(embeddingWeights, freeze=True,
                                                      padding_idx=embeddingDict["PAD"])

        # recurrent unit
        self.rnn = nn.LSTM(self.n_dim, hidden_size, rnn_layers, batch_first=True, dropout=0.5)

        # fully connexted layer
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, mlp_layer_widths)
        self.fc2 = nn.Linear(mlp_layer_widths, 5)

    #pass data through all layers defined
    def forward(self, x_padded, phrase_lengths):
        x_embedded = self.embedding(x_padded)
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x_embedded,
                                                           phrase_lengths,
                                                           batch_first=True, 
                                                           enforce_sorted=False)
        
        # pass through recurrent unit
        _, (h_n, _) = self.rnn(x_packed)
        
        # since a "pack_padded" object is passed to the rnn, the output at the 
        # last timestep takes into account the different sentence lengths
        out = h_n[-1]

        # fully connected layers
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc2(out)

        return out

# function to save model parameters, which is used in early stopping to save the model
# at the epoch with best validation accuracy, before the model overfits the training data
def save_checkpoint(state, ep, filename='checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    print ("=> Saving a new best at epoch:", ep)
    torch.save(state, filename)  # save checkpoint

def trainModel(embeddings_dict, embeddings_weights, trainData, valData, epochs=25, learning_rate=0.001, 
               batch_size=512, hidden_size=300, rnn_layers=1, mlp_layer_widths=100):
    
    #create the data
    dataLoad = DataLoader(trainData, batch_size=batch_size, shuffle=True,
                          collate_fn=pad_collate)
    valDataLoad = DataLoader(valData, batch_size=len(valData),
                          collate_fn=pad_collate)
    
    # lists to store progress of the model at each epoch
    epoch = 0
    losses, valLosses, accuracy, valAccuracy = [], [], [], []

    #copy weights & dict so original weight matrix isn't corrupted by a training run
    weightsCopy = embeddings_weights.clone().detach()
    dictCopy = embeddings_dict
    # create an instance of the model
    model = LSTMLearner(dictCopy, weightsCopy, hidden_size, rnn_layers, mlp_layer_widths)
    # move the model to the gpu
    model = model.to(device)
    
    # The question specifies using the cross entropy loss function
    loss_function = nn.CrossEntropyLoss()
    # AdamW was chosen as the optimiser as it seemed to work best 
    # with the short amount of exploration done
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad,
                                   model.parameters()), lr=learning_rate)

    for ep in range(epochs):
        total_loss = 0
        correct = 0
        # set model to train mode so dropout and batch normalisation layers work as expected
        model.train()

        for _, ((text_pad, phrase_lengths), labels) in enumerate(dataLoad):
            # move the data to the gpu
            text_pad, phrase_lengths, labels = text_pad.to(device), phrase_lengths.to(device), labels.to(device)
            model.zero_grad()
            out = model(text_pad, phrase_lengths)
            loss = loss_function(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(labels)
            # make predictions using the argmax of the output layer
            _, pred = out.max(1)
            correct += (pred == labels).sum().item()
            
        #after each epoch, collect statistics
        losses.append(total_loss/len(trainData))
        accuracy.append(100*correct/len(trainData))
        # statistics about the validation set
        with torch.no_grad():
            # set model to evaluation mode so dropout is no-longer used and all 
            # nodes of the model help in making the prediction
            model.eval()
            (text_pad, phrase_lengths), labels = next(iter(valDataLoad))
            text_pad, phrase_lengths, labels = text_pad.to(device), phrase_lengths.to(device), labels.to(device)
            out = model(text_pad, phrase_lengths)
            valLosses.append(loss_function(out, labels).item() / len(valData))
            _, pred = out.max(1)
            valAccuracy.append(100*(pred == labels).sum().item()/len(valData))

        #if validation improved, save new best model
        if valAccuracy[-1] == max(valAccuracy):
            save_checkpoint(model.state_dict(), ep)
        epoch += 1

    #clean up
    model = model.to(torch.device("cpu"))
    del text_pad, phrase_lengths, labels, out, _, pred, weightsCopy

    return model, losses, valLosses, accuracy, valAccuracy

def plot_model(accuracy, valAccuracy, losses, valLosses):
    # epoch on which the best validation set accuracy occured
    bestEpoch = np.argmax(valAccuracy)

    # a plot of the loss as a function of epoch, with the epoch at which early 
    # stopping is performed marked with a red line
    plt.figure(1)
    plt.plot(losses, label="training set")
    plt.plot(valLosses, label="Validation set")
    plt.axvline(x=bestEpoch, color="r", label="Early stopping epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.figure(2)
    plt.plot(accuracy, label="training set")
    plt.plot(valAccuracy, label="Validation set")
    plt.axvline(x=bestEpoch, color="r", label="Early stopping epoch")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()


def test_model(model, TestData, accuracy, valAccuracy, print_accuracies=True):
  
    with torch.no_grad():
        testDataLoad = DataLoader(TestData, batch_size=len(TestData),
                            collate_fn=pad_collate)
        ((text_pad, phrase_lengths), labels) = next(iter(testDataLoad))
        
        bestModel = copy.deepcopy(model)
        bestModel.load_state_dict(torch.load("checkpoint.pth.tar", map_location=torch.device("cpu")))

        #set model to evaluation mode so dropout works as intended
        bestModel.eval()
        out = bestModel(text_pad, phrase_lengths)
        _, pred = out.max(1)
    
    # epoch on which the best validation set accuracy occured
    bestEpoch = np.argmax(valAccuracy)
    
    # Training and validation set accuracy at that epoch
    bestAccuracy = {"training set":round(accuracy[bestEpoch], 2), 
                    "validation set":round(valAccuracy[bestEpoch], 2)}

    bestAccuracy["test set"] = round(100*(pred == labels).sum().item()/len(TestData), 2)
    
    if print_accuracies:
        for k, v in bestAccuracy.items():
            print(k, ": ", v)
    
    return bestAccuracy

# comment about class
class SST:
    def __init__(self):
        downloadFile('https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip',
             'trainDevTestTrees_PTB.zip')

        self.y_train, self.X_train = ReadTextFile("./trees/train.txt")
        self.y_val, self.X_val = ReadTextFile("./trees/dev.txt")
        self.y_test, self.X_test = ReadTextFile("./trees/test.txt")

    def embed(self, embeddings_dict):
        self.TrainData = sentenceDataset(self.y_train, self.X_train, embeddings_dict)
        self.ValData = sentenceDataset(self.y_val, self.X_val, embeddings_dict)
        self.TestData = sentenceDataset(self.y_test, self.X_test, embeddings_dict)
    
    def reset(self):
        self.TrainData = None
        self.ValData = None
        self.TestData = None

# comment about 
class Extrinsic_Sentiment_Analysis:
    def __init__(self, SST, input_embedding_dict, input_embedding_weights):
        
        self.X_train = SST.X_train
        self.X_val = SST.X_val
        self.X_test = SST.X_test
        self.input_embedding_dict = input_embedding_dict
        self.input_embedding_weights = input_embedding_weights
        
        self.embedding_dict, self.embedding_weights = reduce_preprocess_embedding(self.X_train, self.X_val, self.X_test, input_embedding_dict, input_embedding_weights)

        SST.embed(self.embedding_dict)
        self.TrainData = SST.TrainData
        self.ValData = SST.ValData
        self.TestData = SST.TestData

    def train(self,epochs=100, learning_rate=0.001, batch_size=512, hidden_size=300, rnn_layers=2, mlp_layer_widths=100):
        """train LSTM model using AdamW optimiser with cross-entropy loss"""
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.mlp_layer_widths = mlp_layer_widths

        self.model, self.losses, self.valLosses, self.accuracy, self.valAccuracy = trainModel(self.embedding_dict, self.embedding_weights, self.TrainData, self.ValData,
                                                                                epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, rnn_layers=rnn_layers, 
                                                                                mlp_layer_widths=mlp_layer_widths)

    def test(self, print_accuracies=True, save_test_acc=True, file_path=None):
        try:
            self.bestAccuracy = test_model(self.model, self.TestData, self.accuracy, self.valAccuracy, print_accuracies=True)
        except:
            raise ValueError('Model not trained')

        with open('{}/{}.txt'.format(file_path,'sentiment_analysis_' + str(datetime.now())), 'w') as out:
            out.write("Test accuracy on SST: {}".format(self.bestAccuracy["test set"]))

        print("Test accuracy: {} saved to {}".format(self.bestAccuracy["test set"], str(file_path)))

    def plot(self):
        try:
            plot_model(self.accuracy, self.valAccuracy, self.losses, self.valLosses)
        except:
            raise ValueError('Model note trained')