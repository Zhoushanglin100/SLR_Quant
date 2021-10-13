from __future__ import print_function
import argparse
import os
import logging
from time import strftime

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models.lenet_nb import Net
from models.lenet import Net as Net2
from models.lenet_nb_act import Net as Net3
from models.lenet_nb_act_bn import Net as Net4

from input_data import MNISTDataLoader as DataLoader
# from input_data import CIFAR10DataLoader as DataLoader
import admm
# from parameters import *


### Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--arch', '-a', metavar='ARCH', default='net1',
                    help='model architecture (default: net1)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--admm-epochs', type=int, default=1, metavar='N',
                    help='number of interval epochs to update admm (default: 1)')
parser.add_argument('--max-step', type=int, default=6000, metavar='N',
                    help='number of max step to train (default: 6000)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', type=float, default=1e-6, metavar='M',
                    help='Adam weight decay (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-dir', type=str, default="./checkpoints", metavar='N',
                    help='Directory to save checkpoints')
parser.add_argument('--save-model-name', type=str, default="quantized_mnist_", metavar='N',
                    help='Model name')
parser.add_argument('--load-model-name', type=str, default=None, metavar='N',
                    help='For loading the model')
parser.add_argument('--admm-quant', action='store_true', default=False,
                    help='Choose admm quantization training')
parser.add_argument('--verbose', action='store_true', default=True,
                    help='whether to report admm convergence condition')
parser.add_argument('--evaluate', action='store_true', default=False,
                    help='Just run inference and evaluate on test dataset')
parser.add_argument('--masked', action='store_true', default=False,
                    help='whether to masked training for admm quantization')
parser.add_argument('--optimizer-type', type=str, default='adam',
                    help="choose optimizer type: [sgd, adam]")
parser.add_argument('--logger', action='store_true', default=False,
                    help='whether to use logger')

# -------------------- SLR Parameter ---------------------------------
parser.add_argument('--reg-lambda', type=float, default=0.01, metavar='M',
                    help='initial rho for all layers')
parser.add_argument('--optimization', type=str, default='savlr',
                    help='optimization type: [savlr, admm]')
parser.add_argument('--M', type=int, default=200, metavar='N',
                    help='SLR parameter M ')
parser.add_argument('--r', type=float, default=0.1, metavar='N',
                    help='SLR parameter r ')
parser.add_argument('--initial-s', type=float, default=0.0001, metavar='N',
                    help='SLR parameter initial stepsize')
# -------------------- SLR Quant Type ---------------------------------
parser.add_argument('--quant-type', type=str, default='ternary',
                    help="define sparsity type: [binary,ternary, fixed]")
parser.add_argument('--num-bits', type=str, default="2", metavar='N',
                    help="If use 'fixed' number bits, please set bit length: [8,4,4,8]")

#############################################################################################################

args, unknown = parser.parse_known_args()
best = 0


def main():
    global best
    if args.logger:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()
        try:
            os.makedirs("logger", exist_ok=True)
        except TypeError:
            raise Exception("Direction not create!")
        # logger.addHandler(logging.FileHandler(strftime('logger/mnist_%m-%d-%Y-%H:%M.log'), 'a'))
        log_name = "mnist_{}".format(args.optimization) + "_{}".format(args.quant_type) + "_{}".format(args.num_bits) + "_{}".format(args.M) + "_{}".format(args.r) + "_{}".format(args.initial_s)
        logger.addHandler(logging.FileHandler(strftime('logger/mnist/'+log_name+'.log'), 'a'))
        global print
        print = logger.info

    print("The config arguments showed as below:")
    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset = DataLoader(args.batch_size, args.test_batch_size, kwargs)
    train_loader = dataset.train_loader
    test_loader = dataset.test_loader

    if args.arch =="net1":
        model = Net().to(device)
    elif args.arch =="net2":
        model = Net2().to(device)
    elif args.arch == "net3":
        model = Net3().to(device)
    elif args.arch == "net4":
        model = Net4().to(device)
    print("Arch name is {}".format(args.arch))
    # print(model)

    # for i, (name, W) in enumerate(model.named_parameters()):
    #     print(name)
        # print(i, "th weight:", name, ", shape = ", W.shape, ", weight.dtype = ", W.dtype)  

    if args.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
    else:
        raise ValueError("The optimizer type is not defined!")


    if args.load_model_name:
        model_path = os.path.join(args.save_dir, args.load_model_name)
        print("Path is:{}".format(model_path))
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            try:
                model.load_state_dict(torch.load(model_path)["net"])
            except:
                print("Can't load model!")
                return
        test(args, model, device, test_loader)

    name_list = ["conv1.weight","conv2.weight", "fc1.weight","fc2.weight"]
    if args.quant_type == "fixed":
        num_bits_list = args.num_bits.split(",")
        num_bits_dict = {}
        if len(num_bits_list) == 1:
            for name in model.state_dict():
                if name in name_list:
                    num_bits_dict[name] = int(num_bits_list[0])

        else:
            i = 0
            for name in model.state_dict():
                if name in name_list:
                    print(name + " " + num_bits_list[i])
                    num_bits_dict[name] = int(num_bits_list[i])
                    i += 1
        model.num_bits = num_bits_dict

    # name_list=["conv2.weight","fc1.weight"]

    if args.evaluate:
        test(args, model, device, test_loader)
        admm.test_sparsity(model)
        ctr=0
        for name, W in model.named_parameters():
                if "weight" in name:
                    weight = W.cpu().detach().numpy()
                print(weight)
                ctr=ctr+1
                if ctr >1:
                    break
        return

    model_path = os.path.join(args.save_dir, 'mnist', args.save_model_name)
    if args.admm_quant:
        print("before training")
        test(args, model, device, test_loader)

        if args.masked:
            model.masks = {}
            for name, W in model.named_parameters():
                if "weight" in name:
                    weight = W.cpu().detach().numpy()
                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    zero_mask = torch.from_numpy(non_zeros).to(device)
                    W = torch.from_numpy(weight).to(device)
                    W.data = W
                    model.masks[name] = zero_mask
                    
        acc_array = []
        for epoch in range(1, args.epochs + 1):
            admm_train(args, model, device, train_loader, optimizer, epoch, name_list)
            acc, best_state, best_model = test_quant(args, model, device, test_loader, name_list)
            acc_array.append(acc)
            if best < acc:
                print("Get a new best test accuracy:{:.2f}%\n".format(acc))
                # model_name = model_path + "_{}".format(args.quant_type) + "_acc_{}".format(best) + ".pt"
                # model_new_name = model_path + "_{}".format(args.quant_type) + "_acc_{}".format(acc) + ".pt"
                
                model_name = model_path + "_{}".format(args.optimization) + "_{}".format(args.quant_type) + "_{}".format(args.num_bits) + "_{}".format(args.M) + "_{}".format(args.r) + "_{}".format(args.initial_s) + ".pt"
                model_new_name = model_path + "_{}".format(args.optimization) + "_{}".format(args.quant_type) + "_{}".format(args.num_bits) + "_{}".format(args.M) + "_{}".format(args.r) + "_{}".format(args.initial_s) + ".pt"

                if os.path.isfile(model_name):
                    os.remove(model_name)
                torch.save(best_state, model_new_name)
                best = acc
                last_model = best_model
            else:
                print("Current best test accuracy:{:.2f}%".format(best))

        print(acc_array)

        print("\n!!!!!!!!!!!!!!!!!! Evaluation Result !!!!!!!!!!!!!!!!!!!")
        test(args, last_model, device, test_loader)
        admm.test_sparsity(last_model)

        print("\nCondition 1: ")
        print(model.condition1)
        print("\nCondition 2: ")
        print(model.condition2)

        # admm.apply_quantization(args, model, device)
        # print("\n!!!!!!!!!! Apply quantization!")
        # test(args, last_model, device, test_loader)
        # admm.test_sparsity(last_model)

    else:
        # normal training
        print("Normal training")

        for epoch in range(1, args.epochs + 1):
            adjust_learning_rate(optimizer, epoch)
            train(args, model, device, train_loader, optimizer, epoch)
            acc = test(args, model, device, test_loader)
            if best < acc:
                best_state = model.state_dict()
                print("Get a new best test accuracy:{:.2f}%\n".format(acc))
                model_name = model_path + "_acc_{}".format(best) + ".pt"
                model_new_name = model_path + "_acc_{}".format(acc) + ".pt"
                if os.path.isfile(model_name):
                    os.remove(model_name)
                torch.save(best_state, model_new_name)
                best = acc
            else:
                print("Current best test accuracy:{:.2f}%".format(best))


# normal training process
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# use admm
def admm_train(args, model, device, train_loader, optimizer, epoch, name_list=None):
    model.train()


    if epoch == 1:
        # inialize Z variable
        print("Start admm training quantized network, quantization type: {}".format(args.quant_type))
        admm.admm_initialization(args, model, device, name_list)
    ctr=0
    total_ce = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        ctr += 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        ce_loss = F.nll_loss(output, target)

        total_ce = total_ce + float(ce_loss.item())
        admm.z_u_update(args, model, device, epoch, batch_idx, name_list)  # update Z and U variables
        ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(args, model, ce_loss)  # append admm losss

        #mixed_loss.backward()
        mixed_loss.backward(retain_graph=True)
        if args.masked:
            for i, (name, W) in enumerate(model.named_parameters()):
                if name in model.masks:
                    W.grad *= model.masks[name]

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print("cross_entropy loss: {}, mixed_loss : {}".format(ce_loss, mixed_loss))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), ce_loss.item()))
    model.ce_prev = model.ce
    model.ce = total_ce / ctr


#def test(args, model, device, test_loader):
#    model.eval()
#    test_loss = 0
#    correct = 0
#    testset_size = len(test_loader.dataset)
#    with torch.no_grad():
#        for data, target in test_loader:
#            data, target = data.to(device), target.to(device)
#            output = model(data)
#            test_loss = F.nll_loss(output, target)  # sum up batch loss
#            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#            correct += pred.eq(target.view_as(pred)).sum().item()

#    test_loss /= testset_size

#    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#        test_loss, correct, testset_size,
#        100. * correct / testset_size))
#    return 100. * correct / testset_size


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (100. * correct / len(test_loader.dataset))



def test_quant(args, model, device, test_loader, name_list=None):

    if args.arch == "net1":
        quantized_model = Net().to(device)
    elif args.arch == "net2":
        quantized_model = Net2().to(device)
    elif args.arch == "net3":
        quantized_model = Net3().to(device)
    elif args.arch == "net4":
        quantized_model = Net4().to(device)
    else:
        raise ValueError("Not support this network type!!!!!")

    quantized_model.alpha = model.alpha
    quantized_model.Q = model.Q
    quantized_model.Z = model.Z
    if hasattr(model, 'num_bits'):
        quantized_model.num_bits=model.num_bits
    quantized_model.load_state_dict(model.state_dict())
    print("Apply quantization!")
    admm.apply_quantization(args, quantized_model, device, name_list)
    # test_loss = 0
    # correct = 0
    # testset_size = len(test_loader.dataset)
    # with torch.no_grad():
    #     for data, target in test_loader:
    #         data, target = data.to(device), target.to(device)
    #         output = quantized_model(data)
    #         test_loss = F.cross_entropy(output, target)  # sum up batch loss
    #         pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    #         correct += pred.eq(target.view_as(pred)).sum().item()
    # test_loss /= testset_size
    #
    # print('\nQuantized Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, testset_size,
    #     100. * correct / testset_size))
    acc=test(args, quantized_model, device, test_loader)
    # return 100. * correct / testset_size, quantized_model.state_dict()
    return acc, quantized_model.state_dict(), quantized_model


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 20 epochs"""
    global print
    lr = args.lr * (0.5 ** (epoch // 20))
    print("learning rate ={}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()