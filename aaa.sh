export dataset=cifar10
export net=vgg16

export cuda_num=$1
export quant_type=$3    # [binary, ternary, fixed]
export ext=$4

export M=250             # [50 100 200 300 400 500] $(seq 50 50 500)
export r=0.3             # [0.05 0.1 0.2 0.3]
export initial_s=0.001     # [0.1 0.01 0.001 0.0001 0.00001]

export lambda=0.001
export epochs=150

export fldr=result/train/test_test
mkdir -p $fldr

if [ "$2" == "slr" ]; then
    WANDB_RUN_ID=$quant_type-slr-$4 CUDA_VISIBLE_DEVICES=$cuda_num python3 main_$dataset\_new.py\
                                                --admm-quant --save-model-name "quantized_${dataset}"\
                                                --load-model-name "base/baseline_${net}.pt" --epochs $epochs\
                                                -a $net --logger --optimization 'savlr'\
                                                --quant-type $quant_type\
                                                --M $M --r $r --initial-s $initial_s\
                                                --reg-lambda $lambda\
                                                --ext $ext\
                                                | tee $fldr/$dataset\_$net\_savlr_$quant_type\_$num_bits\_$M\_$r\_$initial_s\_$ext.txt

elif [ "$2" == "admm" ]; then
    WANDB_RUN_ID=$quant_type-admm-$4 CUDA_VISIBLE_DEVICES=$cuda_num python3 main_$dataset\_new.py\
                                                --admm-quant --save-model-name "quantized_${dataset}"\
                                                --load-model-name "base/baseline_${net}.pt" --epochs $epochs\
                                                -a $net --logger\
                                                --optimization 'admm'\
                                                --reg-lambda $lambda\
                                                --quant-type $quant_type\
                                                --ext $ext\
                                                | tee $fldr/quant_$dataset\_$net\_admm_$quant_type\_$ext.txt
fi