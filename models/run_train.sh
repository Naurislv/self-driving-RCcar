#!/bin/bash

train_path="/home/nauris/Dropbox/coding/self-driving-RCcar/train_dataset/"

for i in 10 20 30 40 70 100 # Epochs
do
    save_prefix="TEST"
    
    echo "Eoch count: $i"
    for d in 0 1 # Dropout
    do
        echo "Dropout: $d"
        for a in "selu" "relu" # Activation
        do
            echo "Activation: $a"
            for n in 0 1 # Batch normalization
            do
                echo "Batch Normalization: $n"
                python Train.py --train_path ${train_path} --output_model model_archive/Test --epochs ${i} --dropout ${d} --activation ${a} --batch_norm ${n}
            done
        done
    done
done

# python Train.py --train_path ${train_path} --output_model model_archive/Test --epochs 5 --dropout 1 --activation selu --batch_norm 0