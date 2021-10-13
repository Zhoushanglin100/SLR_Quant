import torch.nn as nn
import torch.nn.functional as F
from utils import TernarizeConv2d, TernarizeLinear

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1, bias=False)
        self.conv2 = TernarizeConv2d(20, 50, 5, 1, bias=False)
        self.fc1 = TernarizeLinear(4*4*50, 500, bias=False)
        self.fc2 = TernarizeLinear(500, 10, bias=False)

        # self.act = nn.Hardtanh()
        # self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(50)
        self.bn3 = nn.BatchNorm1d(500)
        self.bn4 = nn.BatchNorm1d(10)

        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, 4*4*50)
        x = self.act(self.bn3(self.fc1(x)))
        x = self.bn4(self.fc2(x))
        return F.log_softmax(x, dim=1)