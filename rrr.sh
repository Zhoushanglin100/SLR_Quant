export dataset=cifar10
export net=vgg16

export cuda_num=$1
export quant_type=$3    # [binary, ternary, fixed]
# export num_bits=$4      # Only use under "fixed" [4, 8]
# export r=$5             # [0.05 0.1 0.2 0.3]
# export initial_s=$6     # [0.1 0.01 0.001 0.0001 0.00001]
# export M=$7             # [50 100 200 300 400 500] $(seq 50 50 500)
export update=$4
export ext=tmp6

fldr=result/train/test_tmp3
mkdir -p $fldr

if [ "$2" == "slr" ]; then
    for lambda in 0.001 0.01 0.1
    do
        for r in 0.1 0.3 0.05
        do
            for initial_s in 0.001 0.1 
            do
                for M in 50 100 300 500
                do
                    WANDB_RUN_ID=$quant_type-slr-uplam-$update-$lambda-M-$M-r-$r-s-$initial_s-$ext\
                    CUDA_VISIBLE_DEVICES=$cuda_num\
                    python3 main_$dataset\_new.py\
                                --admm-quant --save-model-name "quantized_${dataset}"\
                                --load-model-name "base/baseline_${net}.pt"\
                                -a $net --optimization 'savlr'\
                                --quant-type $quant_type\
                                --M $M --r $r --initial-s $initial_s\
                                --update-rho $update --init-rho $lambda\
                                --ext $ext\
                                | tee $fldr/$dataset\_$net\_savlr_$quant_type\_uplam_$update\_$lambda\_M_$M\_r_$r\_s_$initial_s\_$ext.txt
                done
            done
        done
    done   
elif [ "$2" == "admm" ]; then
    WANDB_RUN_ID=$quant_type-admm-$ext CUDA_VISIBLE_DEVICES=$cuda_num python3 main_$dataset\_new.py\
                                                --admm-quant --save-model-name "quantized_${dataset}"\
                                                --load-model-name "base/baseline_${net}.pt" --epochs $epochs\
                                                -a $net --logger\
                                                --optimization 'admm'\
                                                --reg-lambda $lambda\
                                                --quant-type $quant_type\
                                                --ext $ext\
                                                | tee $fldr/quant_$dataset\_$net\_admm_$quant_type\_$ext.txt
fi