from __future__ import print_function
import argparse
import os
import logging
from time import strftime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import torchvision.models as tchmodels
from models.resnet import resnet18
from models.vgg import VGG

from input_data import CIFAR10DataLoader as DataLoader

import matplotlib.pyplot as plt

##########################################################

kwargs = {'num_workers': 1, 'pin_memory': True}
dataset = DataLoader(64, 1000, kwargs)
train_loader = dataset.train_loader
test_loader = dataset.test_loader

# model = resnet18().cuda()
model = tchmodels.__dict__["resnet18"](pretrained=True)
model = torch.nn.DataParallel(model).cuda()

ttt = "checkpoints/imagenet_resnet18_1/quantized_imagenet__resnet18_savlr_ternary_2_300_0.1_0.01_acc_80.44599914550781_None.pt"

model_path = ttt

print("Path is:{}".format(model_path))
# try:
#     model.load_state_dict(torch.load(model_path))
# except:
#     ckpt = torch.load(model_path)
#     model.load_state_dict(ckpt["net"])

checkpoint = torch.load(model_path)


fig, axs = plt.subplots(5, 5, figsize=(20, 20), sharex=False, sharey=False)
# fig.suptitle("Weight of "+model_path[57:64]+" Model")

idx = 0
for name, W in model.named_parameters():
    # print(name)
    if ("weight" in name) and ("features" in name) and ("features.0.weight" not in name):
        print(name)
        weight = W.cpu().detach().numpy()
        # print(weight)
        weight_array = weight.reshape(-1)

        # plt.hist(weight_array)
        # plt.gca().set(title=name, ylabel='Frequency');
        # plt.savefig("plot_weight/"+name+".png")

        axs[idx//5, idx%5].hist(weight_array)
        axs[idx//5, idx%5].set_title(name)
        idx += 1
plt.savefig("plot_weight/"+model_path[57:64]+".png")

print("Done!")