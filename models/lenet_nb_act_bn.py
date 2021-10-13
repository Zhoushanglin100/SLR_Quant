import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = TernConv2d(1, 20, 5, 1, bias=False)
        # self.conv1 = nn.Conv2d(1, 20, 5, 1, bias=False)
        self.conv2 = TernConv2d(20, 50, 5, 1, bias=False)
        # self.conv2 = nn.Conv2d(20, 50, 5, 1, bias=False)
        self.fc1 = TernLinear(4*4*50, 500, bias=False)
        # self.fc1 = nn.Linear(4*4*50, 500, bias=False)
        self.fc2 = TernLinear(500, 10, bias=False)
        # self.fc2 = nn.Linear(500, 10, bias=False)

        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(50)
        self.bn3 = nn.BatchNorm1d(500)
        self.bn4 = nn.BatchNorm1d(10)



    def forward(self, x):
        x =  F.relu(self.bn1(self.conv1(x)))
        # x = self.bn1(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x =  F.relu(self.bn2(self.conv2(x)))
        # x = self.bn2(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x =  F.relu(self.bn3(self.fc1(x)))
        # x = self.bn3(self.fc1(x))
        x = self.bn4(self.fc2(x))
        return F.log_softmax(x, dim=1)


class TernConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(TernConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        input = BinActive.apply(input)
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        return out


class TernLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(TernLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        input = BinActive.apply(input)
        out = nn.functional.linear(input, self.weight)
        return out

# def activate(input, r=0.5):
#     input[input>r]=1
#     input[input<-r]=-1
#     return input
#
# class MyAct(nn.Module):
#     def __init__(self):
#         '''
#         Init method.
#         '''
#         super().__init__()  # init the base class
#
#     def forward(self, input):
#         '''
#         Forward pass of the function.
#         '''
#         return activate(input)

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input
