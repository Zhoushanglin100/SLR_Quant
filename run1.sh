export dataset=cifar10
export net=vgg16

export cuda_num=$1
export ext=$2

export lambda=0.001
export epochs=200

export fldr=result/train/test_test
mkdir -p $fldr

for quant_type in "binary" "ternary"
do
    WANDB_RUN_ID=$quant_type-admm-$ext CUDA_VISIBLE_DEVICES=$cuda_num python3 main_$dataset\_new.py\
                                                            --admm-quant --save-model-name "quantized_${dataset}"\
                                                            --load-model-name "base/baseline_${net}.pt" --epochs 100\
                                                            -a $net --logger\
                                                            --optimization 'admm'\
                                                            --update-lambda $update --reg-lambda $lambda\
                                                            --quant-type $quant_type\
                                                            --ext $ext\
                                                            | tee $fldr/quant_$dataset\_$net\_admm_$quant_type\_$ext.txt
done