export dataset=cifar10
export net=vgg16

export cuda_num=$1
export quant_type=$2    # [binary, ternary, fixed]
export ext=$3

# export M=$4             # [50 100 200 300 400 500] $(seq 50 50 500)
export r=$4            # [0.05 0.1 0.2 0.3]

export initial_s=0.001     # [0.1 0.01 0.001 0.0001 0.00001]
export epoch=100

export fldr=result/train/test_test
mkdir -p $fldr

for M in 500
do
    WANDB_RUN_ID=$quant_type-slr-$num_bits-$M-$r-$initial_s-$ext CUDA_VISIBLE_DEVICES=$cuda_num python3 main_$dataset\_new.py\
                                                --admm-quant --save-model-name "quantized_${dataset}"\
                                                --load-model-name "base/baseline_${net}.pt" --epochs $epoch\
                                                -a $net --optimization 'savlr'\
                                                --quant-type $quant_type\
                                                --M $M --r $r --initial-s $initial_s\
                                                --ext $ext\
                                                | tee $fldr/$dataset\_$net\_savlr_fixed\_$num_bits\_$M\_$r\_$initial_s\_$ext.txt
done
