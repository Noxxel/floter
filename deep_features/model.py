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
        self.hidden_dim1 = 200
        self.hidden_dim2 = 100

        self.conv1 = nn.Conv1d(self.input_dim, self.hidden_dim1, 3)
        self.conv2 = nn.Conv1d(self.hidden_dim1, self.hidden_dim2, 3)

        self.batchnorm1 = nn.BatchNorm1d(self.hidden_dim1, momentum=0.9)
        self.batchnorm2 = nn.BatchNorm1d(self.hidden_dim2, momentum=0.9)
        self.maxpool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.hidden_dim2, self.hidden_dim2, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim2, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim2),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim2))

    def forward(self, X):
        out = self.conv1(X.view(self.batch_size,X.shape[2],X.shape[1]))
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.dropout(out) #regularize the model
        
        lstm_out, self.hidden = self.lstm(out.view(self.batch_size,out.shape[2],out.shape[1]))
        
        # Only take the output from the final timestep
        out = self.linear(lstm_out[:,-1].view(self.batch_size, -1))
        out = F.log_softmax(out, dim=1)
        return out

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()
