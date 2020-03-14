#!/bin/sh
module load nvidia/cuda/10.0
# get general model
rm -rf ./params_general_cb.pkl
python train_cnnblstm.py train ../dataset/all_train/ ./params_general_cb.pkl false false
python train_cnnblstm.py test ../dataset/all_test/ ./params_general_cb.pkl false false
