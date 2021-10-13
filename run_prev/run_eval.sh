for f in ./checkpoints/*.pt; do
    filename=${f##*/}
    CUDA_VISIBLE_DEVICES=3 python3 main.py --evaluate --load-model-name $filename -a net3 | tee result/evaluate/$filename.txt
done