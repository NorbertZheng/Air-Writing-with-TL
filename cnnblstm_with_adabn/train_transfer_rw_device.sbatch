#!/bin/sh
# train naive cba
rm -rf ./params_rw_device_P30
python train_cnnblstm_with_adabn.py train ../dataset/rw_device/P302MiPlay/train/ ./params_rw_device_P30 false false
python train_cnnblstm_with_adabn.py test ../dataset/rw_device/P302MiPlay/transfer/ ./params_rw_device_P30 false false
# unsuspected
rm -rf ./params_transfer_rw_device_P30_100_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_device/P302MiPlay/transfer/ 0 100 ./params_rw_device_P30 ./params_transfer_rw_device_P30_100_un false false false
# semi-suspected
rm -rf ./params_transfer_rw_device_P30_30_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_device/P302MiPlay/transfer/ 30 0 ./params_rw_device_P30 ./params_transfer_rw_device_P30_30_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_device_P30_50_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_device/P302MiPlay/transfer/ 50 0 ./params_rw_device_P30 ./params_transfer_rw_device_P30_50_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_device_P30_100_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_device/P302MiPlay/transfer/ 100 0 ./params_rw_device_P30 ./params_transfer_rw_device_P30_100_semi false false false

