import torch
import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F

from models.lenet_nb_act_bn import Net
from input_data import MNISTDataLoader as DataLoader

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--save-dir', type=str, default="./checkpoints", metavar='N',
                    help='Directory to save checkpoints')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
parser.add_argument('--load-model-name', type=str, default="quant_wa_mnist_fixed_acc_99.1.pt", metavar='N',
                    help='For loading the model')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
device = torch.device("cuda" if use_cuda else "cpu")
dataset = DataLoader(args.batch_size, args.test_batch_size, kwargs)

test_loader = dataset.test_loader

model = Net()
model_path = os.path.join(args.save_dir, args.load_model_name)
print(model_path)
try:
    state=torch.load(model_path)
except:
    try:
        model = torch.nn.DataParallel(model)
        state=torch.load(model_path)['net']
    except:
        print("Can't load model")
        exit()
#keys=list(state.keys())
#for key in keys:
#
#    state[key[7:]]=state.pop(key)

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    testset_size = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = F.cross_entropy(output, target)  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= testset_size

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, testset_size,
        100. * correct / testset_size))
    return 100. * correct / testset_size


model.load_state_dict(state)
model=model.cuda()
test(args, model, device, test_loader)

import admm

for name, weight in model.named_parameters():
    if "weight" in name:
        print(name)
        unique, counts = np.unique((weight.cpu().detach().numpy()).flatten(), return_counts=True)
        un_list = np.asarray((unique, counts)).T
        # print("Unique quantized weights counts:\n", un_list)
        print("Unique quantized weights length, total levels: ",len(un_list))
        # print(weight.view(weight.numel())[:20])
        #
admm.test_sparsity(model)
# print(model)