#!/bin/sh
module load nvidia/cuda/10.0
# get general model
rm -rf ./params_general_cnn.pkl
python train_naive_cnn.py train ../dataset/all_train/ ./params_general_cnn.pkl
python train_naive_cnn.py test ../dataset/all_test/ ./params_general_cnn.pkl
