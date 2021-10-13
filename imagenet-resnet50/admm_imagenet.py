import argparse
import os
import random
import shutil
import time
from time import strftime
import logging
import warnings
import sys
import torch

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
#from alexnet_bn import AlexNet_BN
#from alexnetNoBN import AlexNetNoBN
#from alexnet import alexnet
#from vgg import vgg16_bn, vgg19_bn
from resnet import ResNet18, ResNet50
#from mobilenetV2 import mobilenet_v2
import yaml
import numpy as np
from testers import *
from parameters import *

import pickle

# sys.path.append('../../')  # append root directory

import admm


import logging
LOG_FILENAME = 'output.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)


#os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = None


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-data', metavar='DIR', default ='/home/guest/data/imagenet/imagenet',#'/home/guest/data/imagenet/imagenet'
                    help='path to dataset')
#parser.add_argument('--logger', action='store_true', default=False,
#                    help='whether to use logger')
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                     # choices=model_names,
#                     help='model architecture: ' +
#                          ' | '.join(model_names) +
#                          ' (default: resnet18)')
parser.add_argument('--arch', type=str, default='resnet',
                    help='[vgg, resnet, convnet, alexnet]')
parser.add_argument('--depth', default=50, type=int,
                    help='depth of the neural network, 16,19 for vgg; 18, 50 for resnet')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr-decay', type=int, default=30, metavar='LR_decay',
                    help='how many every epoch before lr drop (default: 30)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='', type=str,
                    help='url used to set up distributed training')  #'tcp://224.66.41.62:23456'
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--config-file', type=str, default='config_res50',
                    help="define config file")
parser.add_argument('--save-model', type=str, default="pretrained_mnist.pt",
                    help='For Saving the current Model')
parser.add_argument('--load-model', type=str, default="pretrained_mnist.pt",
                    help='For loading the model')
parser.add_argument('--masked-retrain', action='store_true', default=False,
                    help='for masked retrain')
parser.add_argument('--verify', action='store_true', default=False,
                    help='verify model sparsity and accuracy')
parser.add_argument('--verbose', action='store_true', default=True,
                    help='whether to report admm convergence condition')
parser.add_argument('--admm', action='store_true', default=False,
                    help="for admm training")

parser.add_argument('--admm-epoch', type=int, default=3,
                    help="how often we do admm update")
parser.add_argument('--rho', type=float, default=0.1,
                    help="define rho for ADMM")
parser.add_argument('--rho-num', type=int, default = 1,
                    help ="define how many rohs for ADMM training")
parser.add_argument('--sparsity-type', type=str, default='irregular',
                    help="define sparsity_type: [irregular,column,filter]")
parser.add_argument('--optimizer', type=str, default='adam',
                    help='define optimizer')
parser.add_argument('--lr-scheduler', type=str, default='default',
                    help='define lr scheduler')
parser.add_argument('--combine-progressive', action='store_true', default=False,
                    help="for filter pruning after column pruning")

best_acc1 = 0


def main():
    args = parser.parse_args()


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print("ngpus_per_node: ") 
    print(ngpus_per_node)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    
    logging.info('Optimization: ' + optimization)
    logging.info('Epochs: ' + str(args.epochs))
    logging.info('rho: ' + str(args.rho))
   

    logging.info('SLR stepsize: ' + str(initial_s))

    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print("=> creating model '{}{}'".format(args.arch, args.depth))
    # model = AlexNet_BN()
    # model = vgg16_bn()
    # print(model)

    if args.arch == "vgg":
        if args.depth == 16:
            model = vgg16_bn()
        elif args.depth == 19:
            model = vgg19_bn()
        else:
            sys.exit("vgg doesn't have those depth!")
    elif args.arch == "resnet":
        if args.depth == 18:
            #model = ResNet18()
            model = models.resnet18(pretrained=False)
        elif args.depth == 50:
            #model = ResNet50()
            model = models.resnet50(pretrained=True, progress=True)
        else:
            sys.exit("resnet doesn't implement those depth!")
    elif args.arch == "mobilenet":
        args.depth = 2
        model = mobilenet_v2()
    print(model)


    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            print("1")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet'):
            model = torch.nn.DataParallel(model)
            model.cuda()
            print("2")
        else:
            model = torch.nn.DataParallel(model).cuda()
            print("3")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    optimizer = None
    if (args.optimizer == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif (args.optimizer == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), args.lr)


    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader),
                                                         eta_min=4e-08)
    elif args.lr_scheduler == 'default':
        # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
        epoch_milestones = [75, 150, 300, 375, 450]

        """Set the learning rate of each parameter group to the initial lr decayed
            by gamma once the number of epoch reaches one of the milestones
        """
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[i * len(train_loader) for i in epoch_milestones],
                                                   gamma=0.1)
    else:
        raise Exception("unknown lr scheduler")

    """====================="""
    """ multi-rho admm train"""
    """====================="""

    initial_rho = args.rho
    if args.admm:
        # ADMM = admm.ADMM(model, config)
        # admm.admm_initialization(args, ADMM=ADMM, model=model)  # intialize Z variable

        admm_global_epoch = 1
        for i in range(args.rho_num):
            current_rho = initial_rho * 10 ** i
            if i == 0:
                # model.load_state_dict(torch.load("checkpoint.pth.tar")["state_dict"]) # admm train need basline model
                #model.load_state_dict(torch.load("model/resnet18_acc_89.078.pt"))  # admm train need basline model
                model.cuda()
                print("pretrained model loaded!")
                
            else:
                model.load_state_dict(torch.load("model_prunned/imagenet_{}{}_{}_{}_{}_{}.pt".format(
                    args.arch, args.depth, current_rho / 10, args.config_file, args.optimizer, args.sparsity_type)))
                # model.cuda()
            acc = validate(val_loader, model, criterion, args)
            print("Before training starts, accuracy:")
            print(acc)
            logging.info('Initial model accuracy: ' + str(acc))
            ADMM = admm.ADMM(model, file_name="profile/" + args.config_file + ".yaml", rho=current_rho)
            admm.admm_initialization(args, ADMM=ADMM, model=model)  # intialize Z variable

            best_prec1 = 0.

            mixed_losses = []
            accuracy = []
            ce_loss = []
            for epoch in range(1, args.epochs + 1):
                if args.distributed:
                    train_sampler.set_epoch(epoch)

                # admm.admm_adjust_learning_rate(optimizer, admm_global_epoch, args)
                # if args.lr_scheduler == 'default':
                #     adjust_learning_rate(optimizer, admm_global_epoch, args)
                # elif args.lr_scheduler == 'cosine':
                #     scheduler.step()
                #
                #
                # print("current rho: {}\t admm global epoch: {}".format(current_rho, admm_global_epoch))
                # train(train_loader, ADMM, model, criterion, optimizer, epoch, args)
                # prec1 = validate(val_loader, model, criterion, args)
                # best_prec1 = max(prec1, best_prec1)
                # admm_global_epoch += 1

                print("current rho: {}".format(current_rho))
                idx_loss_dict, mixed_loss, loss = train(train_loader, ADMM, model, criterion, optimizer, scheduler, epoch, args)
                prec1 = validate(val_loader, model, criterion, args)
                best_prec1 = max(prec1, best_prec1)

                ce_loss.append(loss)
                mixed_losses.append(mixed_loss)
                accuracy.append(prec1)

            mixed_losses = np.array(mixed_losses)
            ce_loss = np.array(ce_loss)
            accuracy = np.array(accuracy)
            
            
            f = open("results/ADMM_loss{}.pkl".format(current_rho),"wb")
            pickle.dump(ADMM.admmloss,f)
            f.close()
  
            f = open("results/plotable/convergence_wz_{}.pkl".format(current_rho),"wb")
            pickle.dump(ADMM.conv_wz,f)
            f.close()

            f = open("results/plotable/convergence_zz_{}.pkl".format(current_rho),"wb")
            pickle.dump(ADMM.conv_zz,f)
            f.close()


            f = open("results/mixed_losses{}.pkl".format(current_rho),"wb")
            pickle.dump(mixed_losses,f)
            f.close()

            f = open("results/accuracy{}.pkl".format(current_rho),"wb")
            pickle.dump(accuracy,f)
            f.close()

            f = open("results/ce_loss{}.pkl".format(current_rho),"wb")
            pickle.dump(ce_loss,f)
            f.close()

            
            print("Best Acc: {:.4f}".format(best_prec1))
            print("Saving model...")
            torch.save(model.state_dict(), "model_prunned/imagenet_{}{}_{}_{}_{}_{}.pt".format(
                args.arch, args.depth, current_rho, args.config_file, args.optimizer, args.sparsity_type))

    """========================"""
    """END multi-rho admm train"""
    """========================"""



    """=============="""
    """masked retrain"""
    """=============="""

    if args.masked_retrain:
        # load admm trained model
        print("\n>_ Loading file: model_prunned/imagenet_{}{}_{}_{}_{}_{}.pt".format(
            args.arch, args.depth, initial_rho * 10 ** (args.rho_num - 1), args.config_file, args.optimizer, args.sparsity_type))
        model.load_state_dict(torch.load("model_prunned/imagenet_{}{}_{}_{}_{}_{}.pt".format(
            args.arch, args.depth, initial_rho * 10 ** (args.rho_num - 1), args.config_file, args.optimizer, args.sparsity_type)))
        model.cuda()

        ADMM = admm.ADMM(model, file_name="profile/" + args.config_file + ".yaml", rho=initial_rho)
        best_prec1 = [0]

        acc = validate(val_loader, model, criterion, args)
        print("Before hard prune, accuracy:")
        print(acc)
        logging.info('Before hard prune, accuracy: ' + str(acc))

        admm.hard_prune(args, ADMM, model)
        compression = admm.test_sparsity(args,ADMM, model)
        logging.info('Compression rate: ' + str(compression))

        acc = validate(val_loader, model, criterion, args)
        print("After hard prune, accuracy:")
        print(acc)
        logging.info('After hard prune, accuracy ' + str(acc))

        epoch_loss_dict = {}
        testAcc = []

        for epoch in range(1, args.epochs + 1):
            idx_loss_dict = train(train_loader, ADMM, model, criterion, optimizer, scheduler, epoch, args)
            prec1 = validate(val_loader, model, criterion, args)

            if prec1 > max(best_prec1):
                print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(prec1))
                torch.save(model.state_dict(),
                           "model_retrained/imagenet_{}{}_retrained_acc_{:.3f}_{}rhos_{}_{}.pt".format(
                               args.arch, args.depth, prec1, args.rho_num, args.config_file, args.sparsity_type))
                print("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(max(best_prec1)))
                if len(best_prec1) > 1:
                    os.remove("model_retrained/imagenet_{}{}_retrained_acc_{:.3f}_{}rhos_{}_{}.pt".format(
                        args.arch, args.depth, max(best_prec1), args.rho_num, args.config_file, args.sparsity_type))

            epoch_loss_dict[epoch] = idx_loss_dict
            testAcc.append(prec1)

            best_prec1.append(prec1)
            print("\ncurrent best acc is: {:.3f}\n".format(max(best_prec1)))

        test_column_sparsity(model)
        test_filter_sparsity(model)

        print("Best Acc: {:.4f}".format(max(best_prec1)))
        np.save("results/plotable_{}.npy".format(args.sparsity_type), epoch_loss_dict)
        np.save("results/plotable_testAcc_{}.npy".format(args.sparsity_type), testAcc)

    """=================="""
    """end masked retrain"""
    """=================="""


def train(train_loader, ADMM, model, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    total_ce = 0
    ctr = 0 
    idx_loss_dict = {}

    if args.masked_retrain and not args.combine_progressive:
        print("full acc re-train masking")
        masks = {}
        for name, W in (model.named_parameters()):
            # if name not in ADMM.prune_ratios:
            #     continue
            # above_threshold, W = admm.weight_pruning(args, W, ADMM.prune_ratios[name])
            # W.data = W
            # masks[name] = above_threshold
            weight = W.cpu().detach().numpy()
            non_zeros = weight != 0
            non_zeros = non_zeros.astype(np.float32)
            zero_mask = torch.from_numpy(non_zeros).cuda()
            W = torch.from_numpy(weight).cuda()
            W.data = W
            masks[name] = zero_mask
    elif args.combine_progressive:
        print("progressive admm-train/re-train masking")
        masks = {}
        for name, W in (model.named_parameters()):
            weight = W.cpu().detach().numpy()
            non_zeros = weight != 0
            non_zeros = non_zeros.astype(np.float32)
            zero_mask = torch.from_numpy(non_zeros).cuda()
            W = torch.from_numpy(weight).cuda()
            W.data = W
            masks[name] = zero_mask

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        ctr += 1
        mixed_loss_sum = []
        loss = []

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        if args.admm:
            admm.admm_adjust_learning_rate(optimizer, epoch, args)
        else:
            scheduler.step()

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        data = input
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        ce_loss = criterion(output, target)
        total_ce = total_ce + float(ce_loss.item())

        if args.admm:
            # admm.admm_update(args, ADMM, model, device, train_loader, optimizer, epoch, data, i)  # update Z and U
            admm.z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, input, i, writer)  # update Z and U variables
            ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(args, ADMM, model, ce_loss)  # append admm losss
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(ce_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))



        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.admm:
            #mixed_loss.backward()
            mixed_loss.backward(retain_graph=True)
        else:
            ce_loss.backward()

        if args.combine_progressive:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks:
                        W.grad *= masks[name]
        if args.masked_retrain:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks:
                        W.grad *= masks[name]


        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print(
                  'Epoch: [{0}][{1}/{2}] [({3}) lr={4}]\t'
                  'Status: admm-[{5}] retrain-[{6}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), args.optimizer, current_lr, args.admm, args.masked_retrain,
                      batch_time=batch_time, loss=losses, top1=top1, top5=top5))
            print("cross_entropy loss: {}".format(ce_loss))

        if i % 10 == 0:
            idx_loss_dict[i] = top5.avg
        
        if args.admm:
            mixed_loss_sum.append(float(mixed_loss))

        loss.append(float(ce_loss))

    if args.admm:
        lossadmm = []
        for k, v in admm_loss.items():
                #print("at layer {}, admm loss is {}".format(k, v))
                ADMM.admmloss[k].extend([float(v)])
    

    ADMM.ce_prev = ADMM.ce
    ADMM.ce = total_ce / ctr

    return idx_loss_dict, mixed_loss_sum, loss


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top5.avg


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar', ADMM=None):
    filename = 'checkpoint_gpu{}.pth.tar'.format(args.gpu)

    if args.admm and ADMM:  ## add admm variables to state
        state['admm']['ADMM_U'] = ADMM.ADMM_U
        state['admm']['ADMM_Z'] = ADMM.ADMM_Z

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

