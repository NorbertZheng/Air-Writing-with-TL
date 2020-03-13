#!/bin/sh
module load nvidia/cuda/10.0
# train naive cba
rm -rf ./params_rw_people_zh
python train_cnnblstm_with_adabn.py train ../dataset/rw_people/zh2zyt/train/ ./params_rw_people_zh false false
python train_cnnblstm_with_adabn.py test ../dataset/rw_people/zh2zyt/transfer/ ./params_rw_people_zh false false
# unsuspected
rm -rf ./params_transfer_rw_people_zh_100_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_people/zh2zyt/transfer/ 0 100 ./params_rw_people_zh ./params_transfer_rw_people_zh_100_un false false false
# semi-suspected
rm -rf ./params_transfer_rw_people_zh_10_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_people/zh2zyt/transfer/ 10 0 ./params_rw_people_zh ./params_transfer_rw_people_zh_10_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_people_zh_20_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_people/zh2zyt/transfer/ 20 0 ./params_rw_people_zh ./params_transfer_rw_people_zh_20_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_people_zh_30_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_people/zh2zyt/transfer/ 30 0 ./params_rw_people_zh ./params_transfer_rw_people_zh_30_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_people_zh_40_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_people/zh2zyt/transfer/ 40 0 ./params_rw_people_zh ./params_transfer_rw_people_zh_40_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_people_zh_50_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_people/zh2zyt/transfer/ 50 0 ./params_rw_people_zh ./params_transfer_rw_people_zh_50_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_people_zh_60_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_people/zh2zyt/transfer/ 60 0 ./params_rw_people_zh ./params_transfer_rw_people_zh_60_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_people_zh_70_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_people/zh2zyt/transfer/ 70 0 ./params_rw_people_zh ./params_transfer_rw_people_zh_70_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_people_zh_80_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_people/zh2zyt/transfer/ 80 0 ./params_rw_people_zh ./params_transfer_rw_people_zh_80_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_people_zh_90_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_people/zh2zyt/transfer/ 90 0 ./params_rw_people_zh ./params_transfer_rw_people_zh_90_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_people_zh_100_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_people/zh2zyt/transfer/ 100 0 ./params_rw_people_zh ./params_transfer_rw_people_zh_100_semi false false false

