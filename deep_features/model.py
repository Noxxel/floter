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
        self.hidden_dim3 = 64
        self.hidden_dim4 = 128
        self.hidden_dim5 = 64
        self.lstm_hidden = 96
        self.linear_dim = 64

        self.conv1 = nn.Conv1d(self.input_dim, self.hidden_dim1, 3)
        self.conv2 = nn.Conv1d(self.hidden_dim1, self.hidden_dim2, 3)
        self.conv3 = nn.Conv1d(self.hidden_dim2, self.hidden_dim3, 3)
        self.conv4 = nn.Conv1d(self.lstm_hidden, self.hidden_dim4, 3)
        self.conv5 = nn.Conv1d(self.hidden_dim4, self.hidden_dim5, 3)

        self.batchnorm1 = nn.BatchNorm1d(self.hidden_dim1, momentum=0.9)
        self.batchnorm2 = nn.BatchNorm1d(self.hidden_dim2, momentum=0.9)
        self.batchnorm3 = nn.BatchNorm1d(self.hidden_dim3, momentum=0.9)
        self.batchnorm4 = nn.BatchNorm1d(self.hidden_dim4, momentum=0.9)
        self.batchnorm5 = nn.BatchNorm1d(self.hidden_dim5, momentum=0.9)
        self.normLSTM = nn.BatchNorm1d(self.lstm_hidden, momentum=0.9)
        self.normLinear = nn.BatchNorm1d(self.linear_dim, momentum=0.9)
        self.maxpool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.convDrop = nn.Dropout(p=0.25)
        self.dropout = nn.Dropout(p=dropout)

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.hidden_dim3, self.lstm_hidden, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(15264, self.linear_dim)
        self.output = nn.Linear(self.linear_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.lstm_hidden).cuda(),
                torch.zeros(self.num_layers, self.batch_size, self.lstm_hidden).cuda())

    def forward(self, X):
        #print(X.shape)
        X = self.conv1(X.view(X.shape[0],X.shape[2],X.shape[1]))
        X = self.relu(X)
        X = self.batchnorm1(X)
        X = self.maxpool(X)
        X = self.convDrop(X)

        X = self.conv2(X)
        X = self.relu(X)
        X = self.batchnorm2(X)
        X = self.maxpool(X)
        X = self.convDrop(X)

        X = self.conv3(X)
        X = self.relu(X)
        X = self.batchnorm3(X)
        X = self.maxpool(X)
        X = self.convDrop(X) #regularize the model
        #print(X.shape)
        
        lstm_out, hidden = self.lstm(X.view(X.shape[0],X.shape[2],X.shape[1]), self.hidden)
        #print(lstm_out.shape)
        """ X = self.relu(lstm_out[:,-1].view(X.shape[0], self.lstm_hidden, -1))
        X = self.normLSTM(X)
        X = self.dropout(X) """
        
        """ X = self.conv4(lstm_out.contiguous().view(lstm_out.shape[0], lstm_out.shape[2], lstm_out.shape[1]))
        X = self.batchnorm4(X)
        X = self.relu(X)
        X = self.maxpool(X)
        X = self.convDrop(X) """
        #print(X.shape)
        """ X = self.conv5(X)
        X = self.batchnorm5(X)
        X = self.relu(X)
        X = self.maxpool(X)
        X = self.convDrop(X) """

        # Only take the output from the final timestep
        X = lstm_out.contiguous().view(X.shape[0], -1)
        X = self.linear(X)
        X = self.relu(X)
        X = self.normLinear(X)
        X = self.output(X)
        X = F.log_softmax(X, dim=1)
        return X

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()
