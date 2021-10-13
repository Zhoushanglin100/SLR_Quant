import torch.nn as nn
import torch.nn.functional as F


# Define neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio = 2
        self.conv1 = nn.Conv2d(in_channels=1, kernel_size=3, out_channels=3,  bias=None)
        self.htanh = nn.Hardtanh()
        self.activation = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(in_channels=3, kernel_size=3, out_channels=25, padding=1, bias=None)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(25)
        self.fc3 = nn.Linear(25 * 4 * 4, 150)
        self.bn3 = nn.BatchNorm1d(150)
        self.fc4 = nn.Linear(150, 10)
        self.bn4 = nn.BatchNorm1d(10)

        # self.logsoftmax = nn.LogSoftmax(dim=1)
        # self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.htanh(x)

        x = self.pool1(x)
        # x = self.htanh(x)
        x = self.conv2(x)
        # x = self.pool2(x)
        x = self.bn2(x)
        x = self.htanh(x)
        x = self.pool2(x)
        # x = self.htanh(x)

        x = x.view(-1, 25 * 4 * 4)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.htanh(x)
        x = self.fc4(x)
        x = self.bn4(x)
        # return x
        return F.log_softmax(x, dim=1)