from __future__ import print_function

import argparse
import logging
import os, time, random
from time import strftime
import warnings
import numpy as np
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torchvision.models as tchmodels

from models.resnet import resnet18
from models.vgg import VGG
### https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/torchvision_models.py

# import admm.admm_v1 as admm
# import admm.admm_v2 as admm
# import admm.admm_v3 as admm

import wandb
wandb.init(project='SLR-ADMM-imagenet', entity='zhoushanglin100')

#############################################################################################################

model_names = sorted(name for name in tchmodels.__dict__
    if name.islower() and not name.startswith("__")
    and callable(tchmodels.__dict__[name]))

########################################################################################

### Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
                    help='model architecture (default: resnet18, vgg16_bn)')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts/resume)')
parser.add_argument('--admm-epochs', type=int, default=1, metavar='N',
                    help='number of interval epochs to update admm (default: 1)')
parser.add_argument('--max-step', type=int, default=6000, metavar='N',
                    help='number of max step to train (default: 6000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-scheduler', type=str, default='cosine',
                    help="[default, cosine]")
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M',
                    help='Optimizer weight decay (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-dir', type=str, default="./checkpoints", metavar='N',
                    help='Directory to save checkpoints')
parser.add_argument('--save-model-name', type=str, default="quantized_imagenet_", metavar='N',
                    help='Model name')
parser.add_argument('--load-model-name', type=str, default=None, metavar='N',
                    help='For loading the model')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='whether to report admm convergence condition')
parser.add_argument('--evaluate', action='store_true', default=False,
                    help='Just run inference and evaluate on test dataset')
parser.add_argument('--masked', action='store_true', default=False,
                    help='whether to masked training for admm quantization')
parser.add_argument('--optimizer-type', type=str, default='sgd',
                    help="choose optimizer type: [sgd, adam]")
parser.add_argument('--logger', action='store_true', default=False,
                    help='whether to use logger')

# -------------------- ImageNet Specific ---------------------------------
parser.add_argument('-data', metavar='DIR', default ='/data/imagenet',
                    help='path to dataset')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='use pre-trained model')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
# parser.add_argument('--adj-lr', action='store_true', default=False,
#                     help='whether adjust LR')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
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

# -------------------- SLR Parameter ---------------------------------
parser.add_argument('--admm-file', type=str, default="v1_1", metavar='N',
                    help='the admm code file')
# parser.add_argument('-u', '--update-btch', type=int, default=100, metavar='N',
#                     help='frequency of updating Z and U')

parser.add_argument('--admm-quant', action='store_true', default=False,
                    help='Choose admm quantization training')
parser.add_argument('--optimization', type=str, default='savlr',
                    help='optimization type: [savlr, admm]')
parser.add_argument('--quant-type', type=str, default='ternary',
                    help="define sparsity type: [binary,ternary, fixed]")
parser.add_argument('--num-bits', type=str, default="2", metavar='N',
                    help="If use fixed number bits, please set bit length. Ex, --num-bits 8,4,4,8")
parser.add_argument('--update-rho', type=int, default=1, metavar='N',
                    help='Choose whether to update initial rho in each iteration, 1-update, 0-not update')
parser.add_argument('--init-rho', type=float, default=1e-3, metavar='M',
                    help='initial rho for all layers')
parser.add_argument('--M', type=int, default=250, metavar='N',
                    help='SLR parameter M ')
parser.add_argument('--r', type=float, default=0.1, metavar='N',
                    help='SLR parameter r ')
parser.add_argument('--initial-s', type=float, default=0.001, metavar='N',
                    help='SLR parameter initial stepsize ')
parser.add_argument('--ext', type=str, default="None", metavar='N',
                    help='extension of file name')

#############################################################################################################

args, unknown = parser.parse_known_args()
wandb.init(config=args)
wandb.config.update(args)

####################################################################################

if args.admm_file == "v1_1":
    import admm.admm_v1_1 as admm
elif args.admm_file == "v5":
    import admm.admm_v5 as admm

####################################################################################


best_acc = 0


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
        main_worker(args, args.gpu, ngpus_per_node)

# ===========================================================================================

def main_worker(args, gpu, ngpus_per_node):
    global best_acc

    if args.logger:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()
        try:
            fldr_name = "logger/imagenet_"+args.arch+"_"+str(args.update_rho)
            os.makedirs(fldr_name, exist_ok=True)
        except TypeError:
            raise Exception("Direction not create!")
        
        log_name = "imagenet_{}".format(args.arch) + "_{}".format(args.optimization)\
                    + "_{}".format(args.quant_type) + "_{}".format(args.num_bits)\
                    + "_{}".format(args.M) + "_{}".format(args.r) + "_{}".format(args.initial_s)\
                    + "_{}".format(args.ext)

        logger.addHandler(logging.FileHandler(strftime(fldr_name+'/'+log_name+'.log'), 'a'))

        logging.info('Optimization: ' + args.optimization)
        logging.info('Epochs: ' + str(args.epochs))
        logging.info('rho: ' + str(args.rho))
        logging.info('SLR stepsize: ' + str(args.initial_s))

        global print
        print = logger.info

    print("The config arguments showed as below:")
    print(args)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # torch.manual_seed(args.seed)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # ------------------------------------------------------
    # if args.arch == "vgg16":
    #     model = VGG('VGG16')
    #     model = torch.nn.DataParallel(model)
    # elif args.arch == "resnet50":
    #     model = models.resnet50(pretrained=True, progress=True)

    ### create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = tchmodels.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = tchmodels.__dict__[args.arch]()
    
    print("\nArch name is {}".format(args.arch))
    
    wandb.watch(model)
    
    # ------------------------------------------------------
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
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet'):
            model = torch.nn.DataParallel(model)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()


    # ------------------------------------------------------
    ### define loss function (criterion) and pptimizer
    
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
    else:
        raise ValueError("The optimizer type is not defined!")

    # ------------------------------------------------------
    ### optionally resume from a checkpoint

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            if args.gpu is not None:
                # best_acc may be from a checkpoint from a different GPU
                best_acc = best_acc.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    # ------------------------------------------------------
    ### Data loading code

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
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

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # ------------------------------------------------------
    ### Load pretrained model
    if args.load_model_name:
        model_path = os.path.join(args.save_dir, args.load_model_name)
        print("Path is:{}".format(model_path))
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            try:
                ckpt = torch.load(model_path)
                model.load_state_dict(ckpt["net"])
            except:
                print("Can't load model!")
                return
        test(args, model, device, test_loader, criterion)


    print("\n---------------------------------------------")
    
    # print(model)
    for i, (name, W) in enumerate(model.named_parameters()):
        print(name)
    print("---------------------------------------------")
    
    # name_list = []
    # if  (args.arch == "resnet18") or (args.arch == "resnet50"):
    #     for i, (name, W) in enumerate(model.named_parameters()):
    #         if ("weight" in name) and ("layer" in name) and ("conv" in name) :
    #             name_list.append(name)
    # elif args.arch == "vgg16":
    #     for i, (name, W) in enumerate(model.named_parameters()):
    #         if ("features" in name) and ("weight" in name):
    #             # print(name)
    #             name_list.append(name)
    #     for ele in ["module.features.0.weight", "module.features.1.weight", "module.features.41.weight"]:
    #         name_list.remove(ele)
    
    ### vgg16
    # name_list = [
    #              # "module.features.0.weight",
    #              "module.features.1.weight", "module.features.3.weight", 
    #              "module.features.4.weight", "module.features.7.weight", "module.features.8.weight", 
    #              "module.features.10.weight", "module.features.11.weight", "module.features.14.weight", 
    #              "module.features.15.weight", "module.features.17.weight", "module.features.18.weight", 
    #              "module.features.20.weight", "module.features.21.weight", "module.features.24.weight", 
    #              "module.features.25.weight", "module.features.27.weight", "module.features.28.weight", 
    #              "module.features.30.weight", "module.features.31.weight", "module.features.34.weight", 
    #              "module.features.35.weight", "module.features.37.weight", "module.features.38.weight", 
    #              "module.features.40.weight", 
    #              "module.features.41.weight"
    #              ]
    if args.arch == "vgg16":
        name_list = [
                    # features.0.weight",
                    "features.2.weight", "features.5.weight", "features.7.weight", 
                    "features.10.weight", "features.12.weight", "features.14.weight", 
                    "features.17.weight", "features.19.weight", "features.21.weight", 
                    "features.24.weight", "features.26.weight", "features.28.weight", 
                    "classifier.0.weight", "classifier.3.weight", 
                    # "classifier.6.weight",
                    ]
    elif (args.arch == "resnet18") or (args.arch == "resnet50"):
        name_list = []
        for i, (name, W) in enumerate(model.named_parameters()):
            if ("layer" in name) and ("conv" in name) and ("weight" in name):
                name_list.append(name)
    print(name_list)
    print("---------------------------------------------\n")

    # ------------------------------------------------------
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
    
    # ------------------------------------------------------

    if args.evaluate:
        test(args, model, device, test_loader, criterion)
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

    # ------------------------------------------------------
    try:
        os.makedirs(args.save_dir+"/imagenet_"+args.arch+"_"+str(args.update_rho), exist_ok=True)
    except TypeError:
        raise Exception("Direction not create!")
    model_path = os.path.join(args.save_dir, "imagenet_"+args.arch+"_"+str(args.update_rho), args.save_model_name)

    if args.admm_quant:
        # print("Before training")
        # acc1, acc5 = test(args, model, device, test_loader, criterion)
        
        # wandb.log({"Accuracy@1": acc1})
        # wandb.log({"Accuracy@5": acc5})

        # if args.masked:
        #     model.masks = {}
        #     for name, W in model.named_parameters():
        #         if "weight" in name:
        #             weight = W.cpu().detach().numpy()
        #             non_zeros = weight != 0
        #             non_zeros = non_zeros.astype(np.float32)
        #             zero_mask = torch.from_numpy(non_zeros).to(device)
        #             W = torch.from_numpy(weight).to(device)
        #             W.data = W
        #             model.masks[name] = zero_mask

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=4e-08)

        acc_array = []
        iteration = [0]

        # --------------------------------------------------
        if args.optimization == 'admm':

            # for epoch in range(1, args.epochs + 1):
            for epoch in range(args.start_epoch, args.epochs + 1):

                print("\n@@@@@@@@@@@@@@@@@@@@@@ epoch: {} \n".format(epoch))

                # admm.admm_adjust_learning_rate(args, optimizer, epoch)
                if args.lr_scheduler == 'default':
                    adjust_learning_rate(args, optimizer, epoch)
                elif args.lr_scheduler == 'cosine':
                    scheduler.step()

                admm_train(args, model, device, train_loader, optimizer, criterion, epoch, iteration, name_list)
                acc1, acc5, best_state, best_model = test_quant(args, model, device, test_loader, criterion, name_list)

                wandb.log({"Accuracy@1": acc1})
                wandb.log({"Accuracy@5": acc5})

                acc_array.append(acc5)

                if best_acc < acc5:
                    print("Get a new best test accuracy:{:.2f}%\n".format(acc5))
                    # model_name = model_path + "_{}".format(args.quant_type) + "_acc_{}".format(best) + ".pt"
                    # model_new_name = model_path + "_{}".format(args.quant_type) + "_acc_{}".format(acc) + ".pt"
                    model_name = model_path + "_{}".format(args.arch) + "_{}".format(args.optimization)\
                                    + "_{}".format(args.quant_type) + "_{}".format(args.num_bits)\
                                    + "_acc_{}".format(best_acc) + "_{}".format(args.ext)\
                                    + ".pt"
                    model_new_name = model_path + "_{}".format(args.arch) + "_{}".format(args.optimization)\
                                        + "_{}".format(args.quant_type) + "_{}".format(args.num_bits)\
                                        + "_acc_{}".format(acc5) + "_{}".format(args.ext)\
                                        + ".pt"

                    if os.path.isfile(model_name):
                        os.remove(model_name)
                    torch.save(best_state, model_new_name)
                    best_acc = acc5
                    last_model = best_model
                else:
                    print("Current best test accuracy:{:.2f}%".format(best_acc))
                wandb.log({"Best Accuracy": best_acc})

            print(acc_array)

            print("\n!!!!!!!!!!!!!!!!!! Evaluation Result !!!!!!!!!!!!!!!!!!!")
            test(args, last_model, device, test_loader, criterion)
            admm.test_sparsity(last_model)


        # --------------------------------------------------
        elif args.optimization == 'savlr':

            epoch = args.start_epoch-1
            # epoch = 0
            while (iteration[0] < 2*(args.epochs+args.start_epoch)) and (epoch <= 2*args.epochs+args.start_epoch+50):
                epoch += 1
                wandb.log({"Hyper/epoch": epoch})

            # for epoch in range(1, args.epochs + 1):
            # for epoch in range(args.start_epoch, args.epochs + 1):    


                print("\n@@@@@@@@@@@@@@@@@@@@@@ epoch: {} \n".format(epoch))

                admm.admm_adjust_learning_rate(args, optimizer, epoch)
                if args.lr_scheduler == 'default':
                    adjust_learning_rate(args, optimizer, epoch)
                elif args.lr_scheduler == 'cosine':
                    scheduler.step()

                admm_train(args, model, device, train_loader, optimizer, criterion, epoch, iteration, name_list)
                acc1, acc5, best_state, best_model = test_quant(args, model, device, test_loader, criterion, name_list)
                
                wandb.log({"Accuracy@1": acc1})
                wandb.log({"Accuracy@5": acc5})
                
                # print("--------------------------")
                # test(args, model, device, test_loader, criterion)
                # print("--------------------------")


                acc_array.append(acc5)

                print("\n@@@@@@@@@@@@@@@@@@@@@@ iteration: {} \n".format(iteration[0]))

                if best_acc < acc5:
                    print("Get a new best test accuracy:{:.2f}%\n".format(acc5))

                    model_name = model_path + "_{}".format(args.arch) + "_{}".format(args.optimization)\
                                    + "_{}".format(args.quant_type) + "_{}".format(args.num_bits)\
                                    + "_{}".format(args.M) + "_{}".format(args.r) + "_{}".format(args.initial_s)\
                                    + "_acc_{}".format(best_acc) + "_{}".format(args.ext)\
                                    + ".pt"
                    model_new_name = model_path + "_{}".format(args.arch) + "_{}".format(args.optimization)\
                                        + "_{}".format(args.quant_type) + "_{}".format(args.num_bits)\
                                        + "_{}".format(args.M) + "_{}".format(args.r) + "_{}".format(args.initial_s)\
                                        + "_acc_{}".format(acc5) + "_{}".format(args.ext)\
                                        + ".pt"

                    if os.path.isfile(model_name):
                        os.remove(model_name)
                    torch.save(best_state, model_new_name)
                    best_acc = acc5
                    last_model = best_model
                else:
                    print("Current best test accuracy:{:.2f}%".format(best_acc))
                wandb.log({"Best Accuracy": best_acc})

            print(acc_array)

            print("\n!!!!!!!!!!!!!!!!!! Evaluation Result !!!!!!!!!!!!!!!!!!!")
            test(args, last_model, device, test_loader, criterion)
            admm.test_sparsity(last_model)

            print("\nCondition 1: ")
            print(model.condition1)
            print("\nCondition 2: ")
            print(model.condition2)

        # admm.apply_quantization(args, model, device)
        # print("Apply quantization!")
        # test(args, model, device, test_loader)

    else:
        # normal training
        print("Normal training")

        for epoch in range(1, args.epochs + 1):
            adjust_learning_rate(optimizer, epoch)
            train(args, model, device, train_loader, optimizer, criterion, epoch)
            acc1, acc5 = test(args, model, device, test_loader)
            if best_acc < acc5:
                best_state = model.state_dict()
                print("Get a new best test accuracy:{:.2f}%\n".format(acc5))
                model_name = model_path + "_acc_{}".format(best_acc) + ".pt"
                model_new_name = model_path + "_acc_{}".format(acc5) + ".pt"
                if os.path.isfile(model_name):
                    os.remove(model_name)
                torch.save(best_state, model_new_name)
                best_acc = acc5
            else:
                print("Current best test accuracy:{:.2f}%".format(best_acc))


# =============================================================================
### normal training process

def train(args, model, device, train_loader, optimizer, criterion, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    ce_losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        # loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

# =============================================================================
### use admm

def admm_train(args, model, device, train_loader, optimizer, criterion, epoch, iteration, name_list=None):
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    ce_losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.train()

    if epoch == 1:
        # inialize Z variable
        print("Start admm training quantized network, quantization type: {}".format(args.quant_type))
        admm.admm_initialization(args, model, device, name_list)
    
    ctr=0
    total_ce = 0
    idx_loss_dict = {}

    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data_time.update(time.time() - end)

        # # adjust learning rate
        # if args.adj_lr:
        #     adjust_learning_rate(optimizer, epoch, args)
        # else:
        #     scheduler.step()

        ctr += 1

        if args.gpu is not None:
            data = data.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)
       
        optimizer.zero_grad()
        output = model(data)
        ce_loss = criterion(output, target)
        total_ce = total_ce + float(ce_loss.item())

        admm.z_u_update(args, model, device, epoch, iteration, batch_idx, name_list)  # update Z and U variables
        ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(args, model, ce_loss)  # append admm losss

        ### measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        ce_losses.update(ce_loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))

        ### write to W&B log
        wandb.log({"train_acc@1": acc1[0]})
        wandb.log({"train_acc@5": acc5[0]})

        mixed_loss.backward(retain_graph=True)
        
        ### update W, Z, U
        optimizer.step()                                                              # update W

        ### write to W&B log
        wandb.log({"Loss/train_loss": ce_loss.item()})
        wandb.log({"Loss/mixed_loss": mixed_loss.item()})

        ### measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print(
                  'Epoch: [{0}][{1}/{2}] [({3}) lr={4}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx, len(train_loader), args.optimizer_type, current_lr,
                      batch_time=batch_time, loss=ce_losses, top1=top1, top5=top5))
            print("cross_entropy loss: {}".format(ce_loss))
        
        if batch_idx % 10 == 0:
            idx_loss_dict[batch_idx] = top5.avg

        # if batch_idx % args.log_interval == 0:
        #     print("cross_entropy loss: {}, mixed_loss : {}".format(ce_loss, mixed_loss))
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), ce_loss.item()))

    model.ce_prev = model.ce
    model.ce = total_ce / ctr



# =============================================================================
### validation

def test(args, model, device, test_loader, criterion):

    batch_time = AverageMeter('Time', ':6.3f')
    test_losses = AverageMeter('Loss', ':.4e')
    top1_test = AverageMeter('Acc@1', ':6.2f')
    top5_test = AverageMeter('Acc@5', ':6.2f')

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        end = time.time()

        for batch_idx, (data, target) in enumerate(test_loader):
            # data, target = data.to(device), target.to(device)

            if args.gpu is not None:
                data = data.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            output = model(data)
            test_loss = criterion(output, target)

            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            
            wandb.log({"Loss/test_loss": test_loss.item()})

            ### measure accuracy and record loss
            acc1_test, acc5_test = accuracy(output, target, topk=(1, 5))
            test_losses.update(test_loss.item(), data.size(0))
            top1_test.update(acc1_test[0], data.size(0))
            top5_test.update(acc5_test[0], data.size(0))

            ### measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                    batch_idx, len(test_loader), batch_time=batch_time, loss=test_losses,
                                    top1=top1_test, top5=top5_test))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1_test, top5=top5_test))

        wandb.log({"test_acc@1": top1_test.avg})
        wandb.log({"test_acc@5": top5_test.avg})

    return top1_test.avg, top5_test.avg



def test_quant(args, model, device, test_loader, criterion, name_list=None):

    quantized_model = tchmodels.__dict__[args.arch]()
    quantized_model = torch.nn.DataParallel(quantized_model).to(device)

    quantized_model.alpha = model.alpha
    quantized_model.Q = model.Q
    quantized_model.Z = model.Z

    if hasattr(model, 'num_bits'):
        quantized_model.num_bits=model.num_bits

    quantized_model.load_state_dict(model.state_dict())

    print("Apply quantization!")
    admm.apply_quantization(args, quantized_model, device, name_list)

    acc_1, acc_5 = test(args, quantized_model, device, test_loader, criterion)

    return acc_1, acc_5, quantized_model.state_dict(), quantized_model

# =============================================================================
### Helping Function

def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 20 epochs"""
    global print
    lr = args.lr * (0.5 ** (epoch // 20))
    # print("learning rate ={}".format(lr))
    wandb.log({"Hyper/lr": lr})

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
