export dataset=cifar10
export net=resnet18

export cuda_num=$1
export quant_type=$3    # [binary, ternary, fixed]
export num_bits=$4      # Only use under "fixed" [4, 8]
# export r=$5             # [0.05 0.1 0.2 0.3]
# export initial_s=$6     # [0.1 0.01 0.001 0.0001 0.00001]
# export M=$7             # [50 100 200 300 400 500] $(seq 50 50 500)
export lambda=0.01

### Build folder
mkdir -p result/train/$dataset\_$net

for r in 0.05 0.1 0.3
do
    for initial_s in 0.001
    do
        for M in 250 500
        do
            if [ "$2" == "1" ]; then
            ### Ternary and binary
                CUDA_VISIBLE_DEVICES=$cuda_num python3 main_$dataset.py --admm-quant --save-model-name "quantized_${dataset}"\
                                                            --load-model-name "base/baseline_${net}.pt"\
                                                            -a $net --logger\
                                                            --optimization 'savlr' --quant-type $quant_type --num-bits $num_bits\
                                                            --M $M --r $r --initial-s $initial_s\
                                                            --update-lambda --reg-lambda $lambda\
                                                            | tee result/train/$dataset\_$net/$dataset\_$net\_savlr_$quant_type\_$num_bits\_$M\_$r\_$initial_s.txt

            # CUDA_VISIBLE_DEVICES=$cuda_num python3 main.py --evaluate -a net3 --load-model-name quant_mnist_$optimization\_$quant_type\_$num_bits\_$M\_$r\_$initial_s.pt

            elif [ "$2" == "2" ]; then
            ### fixed
                CUDA_VISIBLE_DEVICES=$cuda_num python3 main_$dataset.py --admm-quant --save-model-name "quantized_${dataset}"\
                                                            --load-model-name "base/baseline_${net}.pt"\
                                                            -a $net\
                                                            --optimization 'savlr' --quant-type 'fixed' --num-bits $num_bits\
                                                            --M $M --r $r --initial-s $initial_s\
                                                            --update-lambda --reg-lambda $lambda\
                                                            | tee result/train/$dataset\_$net/$dataset\_$net\_savlr_fixed\_$num_bits\_$M\_$r\_$initial_s.txt
            fi
        done
    done
done

# CUDA_VISIBLE_DEVICES=$cuda_num python3 main_$dataset.py --admm-quant --save-model-name "quantized_${dataset}" --load-model-name "base/baseline_${net}.pt" -a $net --logger --optimization 'admm' --update-lambda $update --reg-lambda $lambda --quant-type $quant_type --num-bits $num_bits | tee result/train/$dataset\_$net/quant_$dataset\_$net\_admm_$quant_type\_$num_bits.txt


# CUDA_VISIBLE_DEVICES=2 python3 main.py --admm-quant --save-model-name "quantized_mnist" --load-model-name "base/baseline_mnist_acc_99.44.pt" -a net3 --logger --optimization 'savlr' --quant-type 'binary' --M 500 --r 0.1 --initial-s 0.001 | tee result/train/quant_mnist_savlr_ternary_500_0.1_0.001.txt

# CUDA_VISIBLE_DEVICES=2 python3 main_mnist.py --admm-quant --save-model-name "quantized_mnist" --load-model-name "base/baseline_net1.pt" -a net1 --logger --optimization 'admm' --reg-lambda 0.001 --quant-type 'ternary' | tee result/train/quant_mnist_admm_ternary.txt
# CUDA_VISIBLE_DEVICES=3 python3 main_cifar10.py --admm-quant --save-model-name "quantized_cifar10" --load-model-name "base/baseline_resnet18.pt" -a resnet18 --logger --optimization 'admm' --reg-lambda 0.01 --quant-type 'ternary' | tee result/train/quant_cifar10_admm_ternary.txt