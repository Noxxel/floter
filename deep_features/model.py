import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, input_dim, batch_size, output_dim=8, num_layers=2, dropout=0.5):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_dim1 = 256
        self.hidden_dim2 = 128
        self.hidden_dim3 = 128
        self.lstm_hidden = 100

        self.conv1 = nn.Conv1d(self.input_dim, self.hidden_dim1, 3)
        self.conv2 = nn.Conv1d(self.hidden_dim1, self.hidden_dim2, 3)
        self.conv3 = nn.Conv1d(self.hidden_dim2, self.hidden_dim3, 3)

        self.batchnorm1 = nn.BatchNorm1d(self.hidden_dim1, momentum=0.9)
        self.batchnorm2 = nn.BatchNorm1d(self.hidden_dim2, momentum=0.9)
        self.batchnorm3 = nn.BatchNorm1d(self.hidden_dim3, momentum=0.9)
        self.batchnorm4 = nn.BatchNorm1d(self.lstm_hidden, momentum=0.9)
        self.maxpool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.hidden_dim3, self.lstm_hidden, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.lstm_hidden, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.lstm_hidden).cuda(),
                torch.zeros(self.num_layers, self.batch_size, self.lstm_hidden).cuda())

    def forward(self, X):
        X = self.conv1(X.view(X.shape[0],X.shape[2],X.shape[1]))
        X = self.batchnorm1(X)
        X = self.relu(X)
        X = self.maxpool(X)
        X = self.dropout(X)

        X = self.conv2(X)
        X = self.batchnorm2(X)
        X = self.relu(X)
        X = self.maxpool(X)
        X = self.dropout(X)

        X = self.conv3(X)
        X = self.batchnorm3(X)
        X = self.relu(X)
        X = self.maxpool(X)
        X = self.dropout(X) #regularize the model
        #print(X.shape)
        
        lstm_out, _ = self.lstm(X.view(X.shape[0],X.shape[2],X.shape[1]), self.hidden)
        #print(lstm_out.shape)
        
        # Only take the output from the final timestep
        X = self.batchnorm4(_[0][0].view(X.shape[0], self.lstm_hidden, -1))
        X = self.dropout(X)
        X = self.linear(X.view(X.shape[0], -1))
        X = F.log_softmax(X, dim=1)
        return X

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()
