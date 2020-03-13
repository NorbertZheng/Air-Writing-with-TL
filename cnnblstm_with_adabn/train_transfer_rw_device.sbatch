#!/bin/sh
module load nvidia/cuda/10.0
# train naive cba
rm -rf ./params_rw_device_P30
python train_cnnblstm_with_adabn.py train ../dataset/rw_device/P302MiPlay/train/ ./params_rw_device_P30 false false
python train_cnnblstm_with_adabn.py test ../dataset/rw_device/P302MiPlay/transfer/ ./params_rw_device_P30 false false
# unsuspected
rm -rf ./params_transfer_rw_device_P30_100_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_device/P302MiPlay/transfer/ 0 100 ./params_rw_device_P30 ./params_transfer_rw_device_P30_100_un false false false
# semi-suspected
rm -rf ./params_transfer_rw_device_P30_10_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_device/P302MiPlay/transfer/ 10 0 ./params_rw_device_P30 ./params_transfer_rw_device_P30_10_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_device_P30_20_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_device/P302MiPlay/transfer/ 20 0 ./params_rw_device_P30 ./params_transfer_rw_device_P30_20_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_device_P30_30_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_device/P302MiPlay/transfer/ 30 0 ./params_rw_device_P30 ./params_transfer_rw_device_P30_30_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_device_P30_40_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_device/P302MiPlay/transfer/ 40 0 ./params_rw_device_P30 ./params_transfer_rw_device_P30_40_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_device_P30_50_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_device/P302MiPlay/transfer/ 50 0 ./params_rw_device_P30 ./params_transfer_rw_device_P30_50_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_device_P30_60_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_device/P302MiPlay/transfer/ 60 0 ./params_rw_device_P30 ./params_transfer_rw_device_P30_60_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_device_P30_70_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_device/P302MiPlay/transfer/ 70 0 ./params_rw_device_P30 ./params_transfer_rw_device_P30_70_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_device_P30_80_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_device/P302MiPlay/transfer/ 80 0 ./params_rw_device_P30 ./params_transfer_rw_device_P30_80_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_device_P30_90_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_device/P302MiPlay/transfer/ 90 0 ./params_rw_device_P30 ./params_transfer_rw_device_P30_90_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_device_P30_100_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_device/P302MiPlay/transfer/ 100 0 ./params_rw_device_P30 ./params_transfer_rw_device_P30_100_semi false false false

