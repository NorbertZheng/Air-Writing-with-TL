#!/bin/sh
# MiPlay_acc_he_slow_hor
module load nvidia/cuda/10.0
rm -rf ./transfer_params5_100_un_MiPlay_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/MiPlay_acc_he_slow_hor/ 0 100 ./params ./transfer_params5_100_un_MiPlay_acc_he_slow_hor false false false
rm -rf ./transfer_params5_10_semi_MiPlay_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/MiPlay_acc_he_slow_hor/ 10 0 ./params ./transfer_params5_10_semi_MiPlay_acc_he_slow_hor false false false
rm -rf ./transfer_params5_20_semi_MiPlay_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/MiPlay_acc_he_slow_hor/ 20 0 ./params ./transfer_params5_20_semi_MiPlay_acc_he_slow_hor false false false
rm -rf ./transfer_params5_30_semi_MiPlay_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/MiPlay_acc_he_slow_hor/ 30 0 ./params ./transfer_params5_30_semi_MiPlay_acc_he_slow_hor false false false
rm -rf ./transfer_params5_40_semi_MiPlay_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/MiPlay_acc_he_slow_hor/ 40 0 ./params ./transfer_params5_40_semi_MiPlay_acc_he_slow_hor false false false
rm -rf ./transfer_params5_50_semi_MiPlay_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/MiPlay_acc_he_slow_hor/ 50 0 ./params ./transfer_params5_50_semi_MiPlay_acc_he_slow_hor false false false
rm -rf ./transfer_params5_60_semi_MiPlay_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/MiPlay_acc_he_slow_hor/ 60 0 ./params ./transfer_params5_60_semi_MiPlay_acc_he_slow_hor false false false
rm -rf ./transfer_params5_70_semi_MiPlay_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/MiPlay_acc_he_slow_hor/ 70 0 ./params ./transfer_params5_70_semi_MiPlay_acc_he_slow_hor false false false
rm -rf ./transfer_params5_80_semi_MiPlay_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/MiPlay_acc_he_slow_hor/ 80 0 ./params ./transfer_params5_80_semi_MiPlay_acc_he_slow_hor false false false
rm -rf ./transfer_params5_90_semi_MiPlay_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/MiPlay_acc_he_slow_hor/ 90 0 ./params ./transfer_params5_90_semi_MiPlay_acc_he_slow_hor false false false
rm -rf ./transfer_params5_100_semi_MiPlay_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/MiPlay_acc_he_slow_hor/ 100 0 ./params ./transfer_params5_100_semi_MiPlay_acc_he_slow_hor false false false
# P30_acc_he_fast_hor
rm -rf ./transfer_params5_100_un_P30_acc_he_fast_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_hor/ 0 100 ./params ./transfer_params5_100_un_P30_acc_he_fast_hor false false false
rm -rf ./transfer_params5_10_semi_P30_acc_he_fast_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_hor/ 10 0 ./params ./transfer_params5_10_semi_P30_acc_he_fast_hor false false false
rm -rf ./transfer_params5_20_semi_P30_acc_he_fast_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_hor/ 20 0 ./params ./transfer_params5_20_semi_P30_acc_he_fast_hor false false false
rm -rf ./transfer_params5_30_semi_P30_acc_he_fast_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_hor/ 30 0 ./params ./transfer_params5_30_semi_P30_acc_he_fast_hor false false false
rm -rf ./transfer_params5_40_semi_P30_acc_he_fast_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_hor/ 40 0 ./params ./transfer_params5_40_semi_P30_acc_he_fast_hor false false false
rm -rf ./transfer_params5_50_semi_P30_acc_he_fast_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_hor/ 50 0 ./params ./transfer_params5_50_semi_P30_acc_he_fast_hor false false false
rm -rf ./transfer_params5_60_semi_P30_acc_he_fast_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_hor/ 60 0 ./params ./transfer_params5_60_semi_P30_acc_he_fast_hor false false false
rm -rf ./transfer_params5_70_semi_P30_acc_he_fast_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_hor/ 70 0 ./params ./transfer_params5_70_semi_P30_acc_he_fast_hor false false false
rm -rf ./transfer_params5_80_semi_P30_acc_he_fast_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_hor/ 80 0 ./params ./transfer_params5_80_semi_P30_acc_he_fast_hor false false false
rm -rf ./transfer_params5_90_semi_P30_acc_he_fast_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_hor/ 90 0 ./params ./transfer_params5_90_semi_P30_acc_he_fast_hor false false false
rm -rf ./transfer_params5_100_semi_P30_acc_he_fast_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_hor/ 100 0 ./params ./transfer_params5_100_semi_P30_acc_he_fast_hor false false false
# P30_acc_he_fast_ver
rm -rf ./transfer_params5_100_un_P30_acc_he_fast_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_ver/ 0 100 ./params ./transfer_params5_100_un_P30_acc_he_fast_ver false false false
rm -rf ./transfer_params5_10_semi_P30_acc_he_fast_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_ver/ 10 0 ./params ./transfer_params5_10_semi_P30_acc_he_fast_ver false false false
rm -rf ./transfer_params5_20_semi_P30_acc_he_fast_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_ver/ 20 0 ./params ./transfer_params5_20_semi_P30_acc_he_fast_ver false false false
rm -rf ./transfer_params5_30_semi_P30_acc_he_fast_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_ver/ 30 0 ./params ./transfer_params5_30_semi_P30_acc_he_fast_ver false false false
rm -rf ./transfer_params5_40_semi_P30_acc_he_fast_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_ver/ 40 0 ./params ./transfer_params5_40_semi_P30_acc_he_fast_ver false false false
rm -rf ./transfer_params5_50_semi_P30_acc_he_fast_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_ver/ 50 0 ./params ./transfer_params5_50_semi_P30_acc_he_fast_ver false false false
rm -rf ./transfer_params5_60_semi_P30_acc_he_fast_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_ver/ 60 0 ./params ./transfer_params5_60_semi_P30_acc_he_fast_ver false false false
rm -rf ./transfer_params5_70_semi_P30_acc_he_fast_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_ver/ 70 0 ./params ./transfer_params5_70_semi_P30_acc_he_fast_ver false false false
rm -rf ./transfer_params5_80_semi_P30_acc_he_fast_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_ver/ 80 0 ./params ./transfer_params5_80_semi_P30_acc_he_fast_ver false false false
rm -rf ./transfer_params5_90_semi_P30_acc_he_fast_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_ver/ 90 0 ./params ./transfer_params5_90_semi_P30_acc_he_fast_ver false false false
rm -rf ./transfer_params5_100_semi_P30_acc_he_fast_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_fast_ver/ 100 0 ./params ./transfer_params5_100_semi_P30_acc_he_fast_ver false false false
# P30_acc_he_slow_hor
rm -rf ./transfer_params5_100_un_P30_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_hor/ 0 100 ./params ./transfer_params5_100_un_P30_acc_he_slow_hor false false false
rm -rf ./transfer_params5_10_semi_P30_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_hor/ 10 0 ./params ./transfer_params5_10_semi_P30_acc_he_slow_hor false false false
rm -rf ./transfer_params5_20_semi_P30_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_hor/ 20 0 ./params ./transfer_params5_20_semi_P30_acc_he_slow_hor false false false
rm -rf ./transfer_params5_30_semi_P30_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_hor/ 30 0 ./params ./transfer_params5_30_semi_P30_acc_he_slow_hor false false false
rm -rf ./transfer_params5_40_semi_P30_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_hor/ 40 0 ./params ./transfer_params5_40_semi_P30_acc_he_slow_hor false false false
rm -rf ./transfer_params5_50_semi_P30_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_hor/ 50 0 ./params ./transfer_params5_50_semi_P30_acc_he_slow_hor false false false
rm -rf ./transfer_params5_60_semi_P30_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_hor/ 60 0 ./params ./transfer_params5_60_semi_P30_acc_he_slow_hor false false false
rm -rf ./transfer_params5_70_semi_P30_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_hor/ 70 0 ./params ./transfer_params5_70_semi_P30_acc_he_slow_hor false false false
rm -rf ./transfer_params5_80_semi_P30_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_hor/ 80 0 ./params ./transfer_params5_80_semi_P30_acc_he_slow_hor false false false
rm -rf ./transfer_params5_90_semi_P30_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_hor/ 90 0 ./params ./transfer_params5_90_semi_P30_acc_he_slow_hor false false false
rm -rf ./transfer_params5_100_semi_P30_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_hor/ 100 0 ./params ./transfer_params5_100_semi_P30_acc_he_slow_hor false false false
# P30_acc_he_slow_ver
rm -rf ./transfer_params5_100_un_P30_acc_he_slow_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_ver/ 0 100 ./params ./transfer_params5_100_un_P30_acc_he_slow_ver false false false
rm -rf ./transfer_params5_10_semi_P30_acc_he_slow_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_ver/ 10 0 ./params ./transfer_params5_10_semi_P30_acc_he_slow_ver false false false
rm -rf ./transfer_params5_20_semi_P30_acc_he_slow_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_ver/ 20 0 ./params ./transfer_params5_20_semi_P30_acc_he_slow_ver false false false
rm -rf ./transfer_params5_30_semi_P30_acc_he_slow_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_ver/ 30 0 ./params ./transfer_params5_30_semi_P30_acc_he_slow_ver false false false
rm -rf ./transfer_params5_40_semi_P30_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_hor/ 40 0 ./params ./transfer_params5_40_semi_P30_acc_he_slow_hor false false false
rm -rf ./transfer_params5_50_semi_P30_acc_he_slow_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_ver/ 50 0 ./params ./transfer_params5_50_semi_P30_acc_he_slow_ver false false false
rm -rf ./transfer_params5_60_semi_P30_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_hor/ 60 0 ./params ./transfer_params5_60_semi_P30_acc_he_slow_hor false false false
rm -rf ./transfer_params5_70_semi_P30_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_hor/ 70 0 ./params ./transfer_params5_70_semi_P30_acc_he_slow_hor false false false
rm -rf ./transfer_params5_80_semi_P30_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_hor/ 80 0 ./params ./transfer_params5_80_semi_P30_acc_he_slow_hor false false false
rm -rf ./transfer_params5_90_semi_P30_acc_he_slow_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_hor/ 90 0 ./params ./transfer_params5_90_semi_P30_acc_he_slow_hor false false false
rm -rf ./transfer_params5_100_semi_P30_acc_he_slow_ver
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_he_slow_ver/ 100 0 ./params ./transfer_params5_100_semi_P30_acc_he_slow_ver false false false
# P30_acc_tong_hor
rm -rf ./transfer_params5_100_un_P30_acc_tong_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_tong_hor/ 0 100 ./params ./transfer_params5_100_un_P30_acc_tong_hor false false false
rm -rf ./transfer_params5_10_semi_P30_acc_tong_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_tong_hor/ 10 0 ./params ./transfer_params5_10_semi_P30_acc_tong_hor false false false
rm -rf ./transfer_params5_20_semi_P30_acc_tong_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_tong_hor/ 20 0 ./params ./transfer_params5_20_semi_P30_acc_tong_hor false false false
rm -rf ./transfer_params5_30_semi_P30_acc_tong_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_tong_hor/ 30 0 ./params ./transfer_params5_30_semi_P30_acc_tong_hor false false false
rm -rf ./transfer_params5_40_semi_P30_acc_tong_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_tong_hor/ 40 0 ./params ./transfer_params5_40_semi_P30_acc_tong_hor false false false
rm -rf ./transfer_params5_50_semi_P30_acc_tong_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_tong_hor/ 50 0 ./params ./transfer_params5_50_semi_P30_acc_tong_hor false false false
rm -rf ./transfer_params5_60_semi_P30_acc_tong_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_tong_hor/ 60 0 ./params ./transfer_params5_60_semi_P30_acc_tong_hor false false false
rm -rf ./transfer_params5_70_semi_P30_acc_tong_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_tong_hor/ 70 0 ./params ./transfer_params5_70_semi_P30_acc_tong_hor false false false
rm -rf ./transfer_params5_80_semi_P30_acc_tong_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_tong_hor/ 80 0 ./params ./transfer_params5_80_semi_P30_acc_tong_hor false false false
rm -rf ./transfer_params5_90_semi_P30_acc_tong_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_tong_hor/ 90 0 ./params ./transfer_params5_90_semi_P30_acc_tong_hor false false false
rm -rf ./transfer_params5_100_semi_P30_acc_tong_hor
python train_transfer_cnnblstm_with_adabn.py ../dataset/all_transfer/P30_acc_tong_hor/ 100 0 ./params ./transfer_params5_100_semi_P30_acc_tong_hor false false false
