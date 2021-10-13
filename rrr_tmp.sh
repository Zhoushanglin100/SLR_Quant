### CIFAR10

# python3 main_cifar10_new.py --admm-quant --load-model-name "base/baseline_vgg16.pt" -a vgg16 --optimization 'savlr' --quant-type ternary --M 500 --r 0.05 --initial-s 0.001 --update-rho 0 --init-rho 0.01 -u 100

# python3 main_cifar10_new.py --admm-quant --load-model-name "base/baseline_vgg16.pt" -a vgg16 --optimization 'savlr' --quant-type binary --M 100 --r 0.3 --initial-s 0.001 --update-rho 1 --init-rho 0.001 -u 100 --admm-file v5 --lr-scheduler default --lr 0.01

### ImageNet

python3 main_imagenet.py --admm-quant --pretrained -a resnet18 --optimization 'savlr' --quant-type ternary --M 500 --r 0.05 --initial-s 0.001 --update-rho 0 --init-rho 0.01 --admm-file v1_1 --lr-scheduler cosine --batch-size 256

python3 main_imagenet.py --admm-quant --pretrained -a resnet18 --optimization 'savlr' --quant-type binary --M 50 --r 0.1 --initial-s 0.001 --update-rho 0 --init-rho 0.1 --admm-file v1_1 --lr-scheduler cosine --batch-size 256