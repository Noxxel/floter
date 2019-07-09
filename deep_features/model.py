import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=8, num_layers=2, dropout=0.5):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        """ self.conv1 = nn.Conv1d(self.input_dim, self.hidden_dim, 128)

        self.batchnorm = nn.BatchNorm1d(self.hidden_dim, momentum=0.9)
        self.maxpool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.dropout2D = nn.Dropout2d(p=dropout) """

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, X):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        #print(X.shape)
        #out = self.conv1(X)
        #print(out.shape)
        #X = nn.utils.rnn.pack_padded_sequence(X enforce_sorted=False)
        #print("0----------------------------------------------------------")
        #print(X.shape)
        lstm_out, self.hidden = self.lstm(X) #.view(len(input), self.batch_size, -1)
        #print("1----------------------------------------------------------")
        #print(lstm_out.shape)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        out = lstm_out[:,-1].reshape(self.batch_size, -1)
        y_pred = self.linear(out)
        y_pred = F.log_softmax(y_pred, dim=1)
        return y_pred #.view(-1)

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()
