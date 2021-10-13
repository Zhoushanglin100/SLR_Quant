#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
#
#python3.6 main.py --evaluate --load-model-name "quantized_mnist_ternary_acc_99.41.pt"
#
#
#python3.6 main.py --evaluate --load-model-name "pruned/lenet_retrain_ckpt_99.2_8.pt"

#python3.6 main.py --evaluate --load-model-name "base/baseline_mnist_acc_99.44.pt"
#
python3 main.py --evaluate --load-model-name "quant_w2a_mnist_fixed_8448_acc_99.15.pt" -a net4
#python3 main.py --evaluate --load-model-name "quant_w3a_mnist_no_bn_fixed_acc_98.88.pt" -a net3
