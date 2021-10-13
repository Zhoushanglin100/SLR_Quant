export dataset=cifar10
export net=vgg16

export cuda_num=$1
export quant_type=$3    # [binary, ternary, fixed]
export num_bits=$4      # Only use under "fixed" [4, 8]
export ext=$5
# export r=$5             # [0.05 0.1 0.2 0.3]
# export initial_s=$6     # [0.1 0.01 0.001 0.0001 0.00001]
# export M=$7             # [50 100 200 300 400 500] $(seq 50 50 500)
export lambda=0.001
export epoch=100

for update in 1
do

    ### Build folder
    # fldr=result/train/$dataset\_$net\_$update
    fldr=result/train/test_tmp3
    mkdir -p $fldr

    for r in 0.005 0.1 0.3
    do
        for initial_s in 0.001
        do
            for M in 250 500
            do
                if [ "$2" == "1" ]; then
                ### Ternary and binary
                    WANDB_RUN_ID=$quant_type-slr-$quant_type-$M-$r-$initial_s-$ext CUDA_VISIBLE_DEVICES=$cuda_num\
                    python3 main_$dataset\_new.py\
                                --admm-quant --save-model-name "quantized_${dataset}"\
                                --load-model-name "base/baseline_${net}.pt" --epochs $epoch\
                                -a $net --optimization 'savlr'\
                                # --logger\
                                --quant-type $quant_type --num-bits $num_bits\
                                --M $M --r $r --initial-s $initial_s\
                                --update-lambda $update --reg-lambda $lambda\
                                --ext $ext\
                                | tee $fldr/$dataset\_$net\_savlr_$quant_type\_$num_bits\_$M\_$r\_$initial_s\_$ext.txt

                # CUDA_VISIBLE_DEVICES=$cuda_num python3 main.py --evaluate -a net3 --load-model-name quant_mnist_$optimization\_$quant_type\_$num_bits\_$M\_$r\_$initial_s.pt

                elif [ "$2" == "2" ]; then
                ### fixed
                    WANDB_RUN_ID=$quant_type-slr-$quant_type-$num_bits-$M-$r-$initial_s-$ext CUDA_VISIBLE_DEVICES=$cuda_num\
                    python3 main_$dataset\_new.py\
                                --admm-quant --save-model-name "quantized_${dataset}"\
                                --load-model-name "base/baseline_${net}.pt" --epochs $epoch\
                                -a $net --optimization 'savlr'\
                                --quant-type 'fixed' --num-bits $num_bits\
                                --M $M --r $r --initial-s $initial_s\
                                --update-lambda $update --reg-lambda $lambda\
                                --ext $ext\
                                | tee $fldr/$dataset\_$net\_savlr_fixed\_$num_bits\_$M\_$r\_$initial_s\_$ext.txt
                fi
            done
        done
    done

    # if [ "$2" == "1" ]; then
    #     WANDB_RUN_ID=$quant_type-admm-$quant_type-$ext CUDA_VISIBLE_DEVICES=$cuda_num\
    #     python3 main_$dataset\_new.py\
    #                 --admm-quant --save-model-name "quantized_${dataset}"\
    #                 --load-model-name "base/baseline_${net}.pt" --epochs 100\
    #                 -a $net --logger\
    #                 --optimization 'admm'\
    #                 --update-lambda $update --reg-lambda $lambda\
    #                 --quant-type $quant_type\
    #                 --ext $ext\
    #                 | tee $fldr/quant_$dataset\_$net\_admm_$quant_type\_$ext.txt
    # elif [ "$2" == "2" ]; then
    #     WANDB_RUN_ID=$quant_type-admm-$quant_type-$ext CUDA_VISIBLE_DEVICES=$cuda_num\
    #     python3 main_$dataset\_new.py\
    #                 --admm-quant --save-model-name "quantized_${dataset}"\
    #                 --load-model-name "base/baseline_${net}.pt" --epochs 100\
    #                 -a $net\
    #                 --optimization 'admm'\
    #                 --update-lambda $update --reg-lambda $lambda\
    #                 --quant-type 'fixed' --num-bits $num_bits\
    #                 --ext $ext\
    #                 | tee $fldr/quant_$dataset\_$net\_admm_$quant_type\_$num_bits\_$ext.txt
    # fi
done

# CUDA_VISIBLE_DEVICES=2 python3 main.py --admm-quant --save-model-name "quantized_mnist" --load-model-name "base/baseline_mnist_acc_99.44.pt" -a net3 --logger --optimization 'savlr' --quant-type 'binary' --M 500 --r 0.1 --initial-s 0.001 | tee result/train/quant_mnist_savlr_ternary_500_0.1_0.001.txt

# CUDA_VISIBLE_DEVICES=0 python3 main_cifar10_new.py --admm-quant --save-model-name "quantized_cifar10" --load-model-name "base/baseline_vgg16.pt" -a vgg16 --logger --optimization 'admm' --update-lambda 1 --reg-lambda 0.001 --quant-type 'binary' | tee result/train/cifar10_vgg16_1/quant_cifar10_vgg16_admm_binary.txt

