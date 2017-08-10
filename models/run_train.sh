#!/bin/bash

python Train.py --train_path /home/nauris/Dropbox/coding/self-driving-RCcar/train_dataset/ --output_model model_archive/nobatch_norm_RELU --epochs 5
python Train.py --train_path /home/nauris/Dropbox/coding/self-driving-RCcar/train_dataset/ --output_model model_archive/nobatch_norm_RELU --epochs 10
python Train.py --train_path /home/nauris/Dropbox/coding/self-driving-RCcar/train_dataset/ --output_model model_archive/nobatch_norm_RELU --epochs 15
python Train.py --train_path /home/nauris/Dropbox/coding/self-driving-RCcar/train_dataset/ --output_model model_archive/nobatch_norm_RELU --epochs 20
python Train.py --train_path /home/nauris/Dropbox/coding/self-driving-RCcar/train_dataset/ --output_model model_archive/nobatch_norm_RELU --epochs 25
python Train.py --train_path /home/nauris/Dropbox/coding/self-driving-RCcar/train_dataset/ --output_model model_archive/nobatch_norm_RELU --epochs 35
python Train.py --train_path /home/nauris/Dropbox/coding/self-driving-RCcar/train_dataset/ --output_model model_archive/nobatch_norm_RELU --epochs 50
python Train.py --train_path /home/nauris/Dropbox/coding/self-driving-RCcar/train_dataset/ --output_model model_archive/nobatch_norm_RELU --epochs 70
python Train.py --train_path /home/nauris/Dropbox/coding/self-driving-RCcar/train_dataset/ --output_model model_archive/nobatch_norm_RELU --epochs 100