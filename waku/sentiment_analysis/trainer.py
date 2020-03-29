import numpy as np
from datetime import datetime
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from waku.sentiment_analysis.model import LSTMLearner
from waku.sentiment_analysis.dataloader import SST
from waku.sentiment_analysis.helpers import reduce_preprocess_embedding, pad_collate


def save_checkpoint(state_dict, filename='checkpoint.pth.tar'):
    """
    Save checkpoint of model used in early stopping 
    
    Parameters:
    -----------
    state_dict : `dict`
        PyTorch dictionary mapping model layer to tensor
    filename : `str`
        The path to save the checkpoints.
    """
    torch.save(state_dict, filename)

def trainModel(embeddings_dict, embeddings_weights, trainData, valData, epochs=25, learning_rate=0.001, 
               batch_size=512, hidden_size=300, rnn_layers=1, mlp_layer_widths=100, checkpoint_filepath='checkpoint.pth.tar'):
    """
    Train LSTMLearner model.

    Parameters:
    -----------
    embeddings_dict : `dict`
        Dict mapping words to int, which indexes row of weight matrix  
    embeddings_weights : `numpy.ndarray``
        (len(vocab), 300) Matrix of word embeddings  
    trainData : `torch.Dataloader`
        Batched training data
    valData : `torch.Dataloader`
        Batched validation data
    epochs : `int`
        Number of epochs for training.
    learning_rate : `float`
        The learning rate for backpropogation.        
    batch_size : `int`
        Number of word pairs in each batch.
    hidden_size : `int`
        Number of features in hidden state of LSTM
    rnn_layers : `int`
        Number of recurrent layers
    mlp_layer_widths : `int`
        Size of fully connected layer
    checkpoint_filepath : `str`
        The path to save the checkpoints.

    Returns:
    --------
    model : `torch.Module`
        Trained LSTMLearner model
    losses : `list` of `float`
        The total training loss in each epoch.
    valLosses : `list` of `float`
        The total validation loss in each epoch.
    accuracy : `list` of `float`
        The training accuracy in each epoch.
    valAccuracy : `list` of `float`
        The validation accuracy in each epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
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
            save_checkpoint(model.state_dict(), checkpoint_filepath)
        epoch += 1

    #clean up
    model = model.to(torch.device("cpu"))
    del text_pad, phrase_lengths, labels, out, _, pred, weightsCopy

    return model, losses, valLosses, accuracy, valAccuracy

def plot_model(accuracy, valAccuracy, losses, valLosses):
    """
    Creates plots for accuracies, losses vs epoch.

    Parameters:
    -----------
    losses : `list` of `float`
        The total training loss in each epoch.
    valLosses : `list` of `float`
        The total validation loss in each epoch.
    accuracy : `list` of `float`
        The training accuracy in each epoch.
    valAccuracy : `list` of `float`
        The validation accuracy in each epoch.
    """
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


def test_model(model, TestData, accuracy, valAccuracy, print_accuracies=False, checkpoint_filepath="checkpoint.pth.tar"):
    """
    Test LSTMLearner model on unseen test data.

    Parameters:
    -----------
    model : `torch.Module`
        Trained LSTMLearner model
    TestData : `torch.Dataloader`
        Batched test data
    accuracy : `list` of `float`
        The training accuracy in each epoch.
    valAccuracy : `list` of `float`
        The validation accuracy in each epoch.
    print_accuracies : `bool`
        Print out dict of training, validation, test accuracies
    checkpoint_filepath : `str`
        The path to load model parameters.

    Returns:
    --------
    bestAccuracy : `dict`, {dataset, accuracy}
        Dictionary of training, validation, test accuracies
    """
    with torch.no_grad():
        # load test data
        testDataLoad = DataLoader(TestData, batch_size=len(TestData),
                            collate_fn=pad_collate)
        ((text_pad, phrase_lengths), labels) = next(iter(testDataLoad))
        
        # load trained model
        bestModel = copy.deepcopy(model)
        bestModel.load_state_dict(torch.load(checkpoint_filepath, map_location=torch.device("cpu")))

        # set model to evaluation mode so dropout works as intended
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


class Extrinsic_Sentiment_Analysis:
    """
    Experiment for sentiment analysis on Stanford Sentinent Treebank
    """
    def __init__(self, SST, input_embedding_dict, input_embedding_weights):
        """
        Arguments:
        ----------
        SST : `Class`
            Data for experiment
        input_embedding_dict : `dict`
            Dict mapping words to int, which indexes row of weight matrix  
        input_embedding_weights : `numpy.ndarray`
            (len(vocab), 300) Matrix of word embeddings
        """
        self.X_train = SST.X_train
        self.X_val = SST.X_val
        self.X_test = SST.X_test
        self.input_embedding_dict = input_embedding_dict
        self.input_embedding_weights = input_embedding_weights
        
        # Reduce embedding to vocab in experiment
        self.embedding_dict, self.embedding_weights = reduce_preprocess_embedding(self.X_train, self.X_val, self.X_test, input_embedding_dict, input_embedding_weights)

        # Process sentences of words into indices which map to embeddings
        SST.embed(self.embedding_dict)
        self.TrainData = SST.TrainData
        self.ValData = SST.ValData
        self.TestData = SST.TestData

    def train(self,epochs=100, learning_rate=0.001, batch_size=512, hidden_size=300, rnn_layers=2, mlp_layer_widths=100, checkpoint_filepath="checkpoint.pth.tar"):
        """
        Train LSTMLearner model.

        Parameters:
        -----------
        epochs : `int`
            Number of epochs for training.
        learning_rate : `float`
            The learning rate for backpropogation.        
        batch_size : `int`
            Number of word pairs in each batch.
        hidden_size : `int`
            Number of features in hidden state of LSTM
        rnn_layers : `int`
            Number of recurrent layers
        mlp_layer_widths : `int`
            Size of fully connected layer
        checkpoint_filepath : `str`
            The path to save the checkpoints.
        """
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.mlp_layer_widths = mlp_layer_widths

        self.model, self.losses, self.valLosses, self.accuracy, self.valAccuracy = trainModel(self.embedding_dict, self.embedding_weights, self.TrainData, self.ValData,
                                                                                epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, rnn_layers=rnn_layers, 
                                                                                mlp_layer_widths=mlp_layer_widths, checkpoint_filepath=checkpoint_filepath)

    def test(self, print_accuracies=False, save_test_acc=False, file_path=None, checkpoint_filepath="checkpoint.pth.tar"):
        """
        Test LSTMLearner model on unseen test data.

        Parameters:
        -----------
        model : `torch.Module`
            Trained LSTMLearner model
        TestData : `torch.Dataloader`
            Batched test data
        accuracy : `list` of `float`
            The training accuracy in each epoch.
        valAccuracy : `list` of `float`
            The validation accuracy in each epoch.
        print_accuracies : `bool`
            Print out dict of training, validation, test accuracies
        save_test_acc : `bool`
            Save results
        file_path : `str`
            Path to save results of experiment
        checkpoint_filepath : `str`
            The path to load model parameters.
        """
        try:
            self.bestAccuracy = test_model(self.model, self.TestData, self.accuracy, self.valAccuracy, print_accuracies=False, checkpoint_filepath=checkpoint_filepath)
        except:
            raise ValueError('Model not trained')

        if save_test_acc:
            with open('{}/{}.txt'.format(file_path,'sentiment_analysis_' + str(datetime.now())), 'w') as out:
                out.write("Test accuracy on SST: {}".format(self.bestAccuracy["test set"]))

        if print_accuracies:
            print("Test accuracy: {}".format(self.bestAccuracy["test set"]))


    def plot(self):
        """
        Plots accurracies and losses
        """
        try:
            plot_model(self.accuracy, self.valAccuracy, self.losses, self.valLosses)
        except:
            raise ValueError('Model not trained')
            
def evaluate(embedding_weights, data_filepath, embedding_dict, checkpoint_filepath):
    """
    Run downstream sentiment analysis experiment on a set of embeddings

    Parameters:
    -----------
    embeddings_weights : `numpy.ndarray`
        (len(vocab), 300) Matrix of word embeddings 
    data_filepath : `str`
        Path for raw Stanford Sentiment Treeback data
    embedding_dict : `dict`
        Dict mapping words to int, which indexes row of weight matrix  
    checkpoint_filepath : `str`
        The path to load model parameters.

    Returns:
    --------
    Experiment.bestAccuracy["test set"] : `float`
        Accuracy on test set
    """
    # Instantiate a *SST* to load Stanford Sentiment Treebank train/test/val data
    SST_instance = SST(data_filepath)

    # Load an instance of the *Extrinsic_Sentiment_Analysis* class with a given dictionary and 
    # weights at which point we reduce the vocabulary into words present in SST
    Experiment = Extrinsic_Sentiment_Analysis(SST_instance, embedding_dict, embedding_weights)

    # Train a specified LSTM model for a given number of epochs using the *train* function
    Experiment.train(epochs=100, learning_rate=0.001, batch_size=512, hidden_size=300, rnn_layers=2, mlp_layer_widths=100, checkpoint_filepath=checkpoint_filepath)

    # Calculate accuracy on the test set and save
    Experiment.test(print_accuracies=False, save_test_acc=False, file_path=None, checkpoint_filepath=checkpoint_filepath)
    
    # reset data class
    SST_instance.reset()
    
    return Experiment.bestAccuracy["test set"]                