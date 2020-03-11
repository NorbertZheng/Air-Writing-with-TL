#!/bin/sh
# train naive cb
rm -rf ./params_rw_device_P30
python train_cnnblstm_with_adabn.py train ../dataset/rw_device/P302MiPlay/train/ ./params_rw_device_P30.pkl false false
python train_cnnblstm_with_adabn.py test ../dataset/rw_device/P302MiPlay/transfer/ ./params_rw_device_P30.pkl false false
