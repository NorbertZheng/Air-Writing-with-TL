#!/bin/sh
module load nvidia/cuda/10.0
# train naive cb
rm -rf ./params_rw_device_P30.pkl
python train_cnnblstm.py train ../dataset/rw_device/P302MiPlay/train/ ./params_rw_device_P30.pkl false false
python train_cnnblstm.py test ../dataset/rw_device/P302MiPlay/transfer/ ./params_rw_device_P30.pkl false false
