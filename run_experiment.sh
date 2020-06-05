#!/bin/bash

# set -x

# train all models on all datasets
# could be time consuming
cd src/;

# train each model  on cifar10
python code/main.py --dataset=CIFAR10 --algorithm=WideResNet28x10 --epochs=300 --batch_size=128 --optimizer=SGD --lr=0.01 --weight_decay=5e-4 --momentum=0.9;
python code/main.py --dataset=CIFAR10 --algorithm=WideResNet28x10Drop --epochs=300 --batch_size=128 --optimizer=SGD --lr=0.01 --weight_decay=5e-4 --momentum=0.9;
python code/main.py --dataset=CIFAR10 --algorithm=WideResNet28x10 --epochs=300 --batch_size=128 --optimizer=SGD --lr=0.01 --weight_decay=5e-4 --momentum=0.9 --swa --swa_start=161 --swa_lr=0.01 --cov_mat;
python code/main.py --dataset=CIFAR10 --algorithm=WideResNet28x10 --method=dpn --epochs=300 --batch_size=128 --optimizer=SGD --lr=0.01 --weight_decay=5e-4 --momentum=0.9;
python code/main.py --dataset=CIFAR10 --algorithm=WideResNet28x10 --method=jem --epochs=300 --batch_size=128 --optimizer=SGD --lr=0.01 --weight_decay=5e-4 --momentum=0.9;

# obtain calibration plots and entropies for in distribution
for dataset in {'CIFAR10'}; do
    for algorithm in {'WideResNet28x10','WideResNet28x10Drop'}; do
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
