export dataset=cifar10
export net=vgg16
export update=1

folder=$dataset\_$net\_$update

exec 3<> $dataset\_$net\_$update\_result.csv

# --------------------------------------------------------------
### Extract Fixed result

# exec 3<> $dataset\_$net\_$update\_fixed_result.csv
echo 'fixed' >&3
echo 'epoch,num_bits,acc,sparsity,compression,M,r,s0' >&3

for f in ./train/$folder/$dataset\_$net\_savlr_fixed_*.txt; do
    filename_full=${f##*/}
    filename=${filename_full%.txt}
    echo $filename_full

    num_bits=$(echo $filename | cut -d '_' -f 5)
    M=$(echo $filename | cut -d '_' -f 6)
    r=$(echo $filename | cut -d '_' -f 7)
    s0=$(echo $filename | cut -d '_' -f 8)
    # echo $num_bits $M $r $s0

    acc=$(cat $f | grep -A 2 'Evaluation Result' | tail | cut -d ' ' -f 8 | cut -c 2-6)
    sparsity=$(cat $f | grep -a 'Total sparsity' | cut -d ' ' -f 3 | cut -d ',' -f 1)
    compression=$(cat $f | grep -a 'Total sparsity' | cut -d ' ' -f 6)
    epoch=$(cat $f | grep -a 'Train Epoch' | tail -1 | cut -d ' ' -f 3)
    # echo $acc $sparsity $compression $epoch

    echo $epoch','$num_bits','$acc','$sparsity','$compression','$M','$r','$s0 >&3
done

# echo >&3
# echo 'epoch,num_bits,acc,sparsity,compression' >&3

# for f_admm in ./train/$folder/quant_$dataset\_$net\_admm_fixed*.txt; do
#     filename_full=${f_admm##*/}
#     filename=${filename_full%.txt}
#     echo $filename_full

#     num_bits=$(echo $filename | cut -d '_' -f 6)   
#     acc=$(cat $f_admm | grep -A 2 'Evaluation Result' | tail | cut -d ' ' -f 8 | cut -c 2-6)
#     epoch=$(cat $f_admm | grep -a 'Train Epoch' | tail -1 | cut -d ' ' -f 3)
#     sparsity=$(cat $f_admm | grep -a 'Total sparsity' | cut -d ' ' -f 3 | cut -d ',' -f 1)
#     compression=$(cat $f_admm | grep -a 'Total sparsity' | cut -d ' ' -f 6)

#     echo $epoch','$num_bits','$acc','$sparsity','$compression >&3

# done

# # exec 3>&-

# -------------------------------------------------------------------------------
### Extract binary result

# exec 3<> $dataset\_$net\_$update\_binary_result.csv
echo >&3
echo 'Binary' >&3
echo 'epoch,num_bits,acc,sparsity,compression,M,r,s0' >&3

for f in ../logger/$folder/$dataset\_$net\_savlr_binary_*.log; do
    filename_full=${f##*/}
    filename=${filename_full%.log}
    echo $filename_full

    M=$(echo $filename | cut -d '_' -f 6)
    r=$(echo $filename | cut -d '_' -f 7)
    s0=$(echo $filename | cut -d '_' -f 8)
    # echo $num_bits $M $r $s0

    acc=$(cat $f | grep -A 2 'Evaluation Result' | tail | cut -d ' ' -f 8 | cut -c 2-6)
    epoch=$(cat $f | grep -a 'Train Epoch' | tail -1 | cut -d ' ' -f 3)
    # echo $acc $epoch

    echo $epoch','','$acc','','','$M','$r','$s0 >&3
done

# echo >&3
# echo 'epoch,acc' >&3

# match_admm=../logger/$folder/$dataset\_$net\_admm_binary*.log
# acc=$(cat $f | grep -A 2 'Evaluation Result' | tail | cut -d ' ' -f 8 | cut -c 2-6)
# epoch=$(cat $f | grep -a 'Train Epoch' | tail -1 | cut -d ' ' -f 3)

# echo $epoch','$acc >&3

# # exec 3>&-

# -------------------------------------------------------------------------------
### Extract ternary result

# exec 3<> $dataset\_$net\_$update\_ternary_result.csv
echo >&3
echo 'Ternary' >&3
echo 'epoch,num_bits,acc,sparsity,compression,M,r,s0' >&3

for f in ../logger/$folder/$dataset\_$net\_savlr_ternary_*.log; do
    filename_full=${f##*/}
    filename=${filename_full%.log}
    echo $filename_full

    M=$(echo $filename | cut -d '_' -f 6)
    r=$(echo $filename | cut -d '_' -f 7)
    s0=$(echo $filename | cut -d '_' -f 8)
    if [ "$s0" == 1e-05 ]; then s0=$(printf "%.5f" $s0); fi
    # echo $num_bits $M $r $s0

    acc=$(cat $f | grep -A 2 'Evaluation Result' | tail | cut -d ' ' -f 8 | cut -c 2-6)
    epoch=$(cat $f | grep -a 'Train Epoch' | tail -1 | cut -d ' ' -f 3)

    match_file=./train/$folder/$dataset\_$net\_savlr_ternary_0_$M\_$r\_$s0.txt
    # echo $match_file

    sparsity=$(cat $match_file | grep -a 'Total sparsity' | cut -d ' ' -f 3 | cut -d ',' -f 1)
    compression=$(cat $match_file | grep -a 'Total sparsity' | cut -d ' ' -f 6)
    # echo $acc $sparsity $compression

    echo $epoch','','$acc','$sparsity','$compression','$M','$r','$s0 >&3
done

# echo >&3
# echo 'epoch,acc,sparsity,compression' >&3

# match_admm=../logger/$folder/$dataset\_$net\_admm_ternary*.log
# acc=$(cat $f | grep -A 2 'Evaluation Result' | tail | cut -d ' ' -f 8 | cut -c 2-6)
# epoch=$(cat $f | grep -a 'Train Epoch' | tail -1 | cut -d ' ' -f 3)

# match_admm_file=./train/$folder/quant_$dataset\_$net\_admm_ternary.txt
# sparsity=$(cat $match_file | grep -a 'Total sparsity' | cut -d ' ' -f 3 | cut -d ',' -f 1)
# compression=$(cat $match_file | grep -a 'Total sparsity' | cut -d ' ' -f 6)

# echo $epoch','$acc','$sparsity','$compression >&3

exec 3>&-

