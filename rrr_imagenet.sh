### Using SLR quantization method
# ternary quant
python3 main_imagenet.py --admm-quant --pretrained -a resnet18 --optimization 'savlr' --quant-type ternary --M 500 --r 0.05 --initial-s 0.001 --update-rho 0 --init-rho 0.01 --admm-file v1_1 --lr-scheduler cosine

# binary quant
python3 main_imagenet.py --admm-quant --pretrained -a resnet18 --optimization 'savlr' --quant-type binary --M 50 --r 0.1 --initial-s 0.001 --update-rho 0 --init-rho 0.1 --admm-file v1_1 --lr-scheduler cosine



### Using ADMM quantization method
# ternary quant
python3 main_imagenet.py --admm-quant --pretrained -a resnet18 --optimization 'admm' --quant-type ternary --update-rho 0 --init-rho 0.01 --admm-file v1_1 --lr-scheduler cosine

# binary quant
python3 main_imagenet.py --admm-quant --pretrained -a resnet18 --optimization 'admm' --quant-type binary --update-rho 0 --init-rho 0.1 --admm-file v1_1 --lr-scheduler cosine
