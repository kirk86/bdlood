#!/bin/bash

# set -x

# train all models on all datasets
# could be time consuming
cd src/;
# for dataset in {'CIFAR10','SVHN'}; do
#     for algorithm in {'PreResNet56','VGG16'}; do
#         for optim in {'SGD',}; do
#             # train models
#             # python -m torch.distributed.launch code/main.py \
#             python code/main.py \
#                    --dataset=$dataset \
#                    --algorithm=$algorithm \
#                    --epochs=5 \
#                    --batch_size=1024 \
#                    --optimizer=$optim \
#                    --lr=0.05 \
#                    --weight_decay=5e-4 \
#                    --momentum=0.9;
#         done
#     done
# done

# obtain calibration plots and entropies for in distribution
for dataset in {'CIFAR10','SVHN'}; do
    for algorithm in {'VGG16','PreResNet56'}; do
        for optim in {'SGD',}; do
            # train models
            # python code/main.py \
            #        --dataset=$dataset \
            #        --algorithm=$algorithm \
            #        --epochs=3 \
            #        --batch_size=1024 \
            #        --optimizer=$optim \
            #        --lr=0.05 \
            #        --weight_decay=5e-4 \
            #        --momentum=0.9;
            # evaluate models, obtain entropies and calibration curves for inclass
            python code/evaluate.py \
                   --algorithm=$algorithm \
                   --dataset=$dataset \
                   --optimizer=$optim;
                   # --chkpt=$algorithm"_"$optim"_"$dataset;
        done
    done
done
