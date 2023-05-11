import torch
import torch.nn as nn
import pandas as pd


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.activation2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.activation3 = nn.Softmax()

    def forward(self, x):
        out = self.l1(x)
        out = self.activation1(out)
        out = self.l2(out)
        out = self.activation2(out)
        out = self.l3(out)
        out = self.activation3(out)
        return out


if __name__ == "__main__":
    device = torch.device('cpu')

    # model = NeuralNetwork(7, 21, 3)
    #
    # criteria = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # input_size = 7
    # hidden_size = 21
    # num_classes = 3
    # num_epochs = 10
    # batch_size = 1000
    # learning_rate = 0.001
