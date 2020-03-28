import torch
import torch.nn as nn
import torch.nn.functional as F

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