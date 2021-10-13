export net=resnet18

export cuda_num=$1
export quant_type=$2
export r=$3             # [0.05 0.1 0.2 0.3]
export initial_s=$4     # [0.1 0.01 0.001 0.0001 0.00001]
export M=$5             # [50 100 200 300 400 500] $(seq 50 50 500)

export update=$6        # [0 1]
export lambda=$7        # [0.001 0.01 0.1]

export ext=imgnet_tmp1

CUDA_VISIBLE_DEVICES=$cuda_num\
python3 main_imagenet.py --admm-quant --pretrained --adj-lr\
                         -a $net --optimization 'savlr'\
                         --quant-type $quant_type\
                         --M $M --r $r --initial-s $initial_s --update-rho $update --init-rho $lambda\
                         --ext $ext