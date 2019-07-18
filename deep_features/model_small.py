import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, input_dim, batch_size, output_dim=8, num_layers=2, dropout=0.4):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.hidden_dim1 = 96
        self.hidden_dim2 = 64  
        self.hidden_dim3 = 32
        self.lstm_hidden = 64
        self.linear_dim = 32
        
        self.conv1 = nn.Conv1d(self.input_dim, self.hidden_dim1, 5, padding=2)
        self.conv2 = nn.Conv1d(self.hidden_dim1, self.hidden_dim2, 5, padding=2)
        self.conv3 = nn.Conv1d(self.hidden_dim2, self.hidden_dim3, 5, padding=2)

        self.batchnorm1 = nn.BatchNorm1d(self.hidden_dim1, momentum=0.9)
        self.batchnorm2 = nn.BatchNorm1d(self.hidden_dim2, momentum=0.9)
        self.batchnorm3 = nn.BatchNorm1d(self.hidden_dim3, momentum=0.9)

        self.lstm = nn.LSTM(self.hidden_dim3, self.lstm_hidden, self.num_layers, batch_first=True)
        self.normLSTM = nn.BatchNorm1d(self.lstm_hidden, momentum=0.9)

        self.linear = nn.Linear(self.lstm_hidden, self.linear_dim)
        self.normLin = nn.BatchNorm1d(self.linear_dim, momentum=0.9)

        self.output = nn.Linear(self.linear_dim, output_dim)
        #self.output = nn.Linear(self.lstm_hidden, output_dim)

        self.maxpool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, device):
        return (torch.zeros(self.num_layers, self.batch_size, self.lstm_hidden).to(device),
                torch.zeros(self.num_layers, self.batch_size, self.lstm_hidden).to(device))

    def convolve(self, X):
        X = self.conv1(X.view(X.shape[0],X.shape[2],X.shape[1]))
        X = self.batchnorm1(X)
        X = self.relu(X)
        X = self.dropout(X)

        X = self.conv2(X)
        X = self.batchnorm2(X)
        X = self.relu(X)
        X = self.dropout(X)

        X = self.conv3(X)
        X = self.batchnorm3(X)
        X = self.relu(X)
    
        return X.view(X.shape[0],X.shape[2],X.shape[1])

    def forward(self, X):
        X = self.convolve(X)
        
        X = self.maxpool(X.view(X.shape[0], X.shape[2], X.shape[1]))
        X = self.dropout(X.view(X.shape[0], X.shape[2], X.shape[1]))

        lstm_out, hidden = self.lstm(X, self.hidden)
        X = self.normLSTM(lstm_out[:,-1].view(X.shape[0], -1))

        X = self.linear(X)
        X = self.normLin(X)

        X = self.output(X)
        X = F.log_softmax(X, dim=1)
        return X

    def get_accuracy(self, logits, target):
        corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()
