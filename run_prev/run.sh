#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

#python3.6 main.py --lr 0.001 --optimizer-type sgd --admm-quant --quant-type ternary --epoch 50 --load-model-name "pruned/lenet_retrain_ckpt_99.2_8.pt" --masked --verbose --logger

#python3.6 main.py --lr 0.01 --optimizer-type sgd --quant-type ternary --epoch 50

#python3.6 main.py --lr 0.001 --optimizer-type adam --epoch 70 --save-model-name "baseline_mnist"

python3 main.py --lr 0.0001 --optimizer-type adam --epoch 60 --admm-quant --save-model-name "quantized_mnist" --load-model-name "base/baseline_mnist_acc_99.44.pt" --verbose --logger -a net3

python3 main.py --optimizer-type sgd --lr 0.01 --epoch 60 --save-model-name "quan_a_mnist_no_bn" --verbose --logger -a net3

#python3 main.py --lr 0.0001 --optimizer-type adam --epoch 60 --admm-quant --quant-type fixed --save-model-name "quant_wa_mnist_no_bn" --load-model-name "base/quan_a_mnist_acc_98.86.pt" --verbose --logger -a net3 --num-bits 8,4,4,8