export dataset=mnist
export net=net1

export cuda_num=$1
export quant_type=$3    # [binary, ternary, fixed]
export num_bits=$4      # Only use under "fixed" [4, 8]

export M=(250 500)             # [50 100 200 300 400 500] $(seq 50 50 500)
export r=(0.05 0.1 0.3)             # [0.05 0.1 0.2 0.3]
export initial_s=(0.001 0.001)    # [0.1 0.01 0.001 0.0001 0.00001]

export lambda=0.001

for subset in subsets
do
    $M=subset[0]
    $r=subset[1]
    $initial_s=subset[2]
    if [ "$2" == "1" ]; then
    ### Ternary and binary
        CUDA_VISIBLE_DEVICES=$cuda_num python3 main_$dataset.py --admm-quant --save-model-name "quantized_${dataset}"\
                                                    --load-model-name "base/baseline_${net}.pt"\
                                                    -a $net --logger\
                                                    --optimization 'savlr' --quant-type $quant_type --num-bits $num_bits\
                                                    --M $M --r $r --initial-s $initial_s\
                                                    --reg-lambda $lambda\
                                                    | tee result/train/$dataset/$dataset\_savlr_$quant_type\_$num_bits\_$M\_$r\_$initial_s.txt

    elif [ "$2" == "2" ]; then
    ### fixed
        CUDA_VISIBLE_DEVICES=$cuda_num python3 main_$dataset.py --admm-quant --save-model-name "quantized_${dataset}"\
                                                    --load-model-name "base/baseline_${net}.pt"\
                                                    -a $net\
                                                    --optimization 'savlr' --quant-type 'fixed' --num-bits $num_bits\
                                                    --M $M --r $r --initial-s $initial_s\
                                                    --reg-lambda $lambda\
                                                    | tee result/train/$dataset/$dataset\_savlr_fixed\_$num_bits\_$M\_$r\_$initial_s.txt
    fi
done

# CUDA_VISIBLE_DEVICES=2 python3 main_mnist.py --admm-quant --save-model-name "quantized_mnist" --load-model-name "base/baseline_net1.pt" -a net1 --logger --optimization 'admm' --reg-lambda 0.001 --quant-type 'ternary' | tee result/train/quant_mnist_admm_ternary.txt
