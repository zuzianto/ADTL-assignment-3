#!/bin/bash

# 定义 GPU 列表


# 1. 第一个循环: seed
for seed in 1233 1234 1235; do
    # 2. 第二个循环: size
    for scale in 4 3 5 6 7; do
        for size in 28 38 48 96 144 192 240 288; do
            
            CSV_NAME="my_select_${size}_scale${scale}_un10.csv"
            OUT_NAME="my_scale${scale}_un10_size${size}_seed${seed}"

            
            (
                export CUDA_VISIBLE_DEVICES=0,1,2,3
                echo "------------------------------------------------"
                
                # 执行训练 (建议将输出重定向，否则3个任务混在一起看不清)
                python train.py \
                    -b 4 \
                    --n_gpu 4 \
                    --root_path "../data/Promise12/Foreslices" \
                    --csv_path "../data/Promise12/Foreslices/splits/${CSV_NAME}" \
                    -o "../output/Promise12/${OUT_NAME}" \
                    --seed "${seed}" \
                    --train_max_epochs 300
                
                # 执行推理
                python inference.py \
                    --adapt_ckpt "../output/Promise12/${OUT_NAME}/mse_best_model.pth" \
                    --root_path "../data/Promise12/Foreslices" \
                    --stage "test" \
                    --num_classes 2 \
                    --img_size 512 \
                    --seed ${seed} \
                    --cal_hd95
            ) 

                
        done
    done
done

wait
echo "All tasks finished successfully."