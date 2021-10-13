import torch.nn.functional as F
from utils import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, 1)
        self.conv2 = nn.Conv2d(3, 25, 3, 1, 1)
        self.fc1 = nn.Linear(4 * 4 * 25, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        x=ternarize(x)
        x = F.tanh(self.conv1(x))
        x = ternarize(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.tanh(self.conv2(x))
        x = ternarize(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 25)
        x = F.tanh(self.fc1(x))
        x = ternarize(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
