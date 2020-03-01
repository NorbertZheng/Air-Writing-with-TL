#!/bin/sh
rm -rf ./transfer_params_25_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_T1/ 0 25 ./params_6dmg_T1 ./transfer_params_6dmg_T1_25_un true
rm -rf ./transfer_params_50_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_T1/ 0 50 ./params_6dmg_T1 ./transfer_params_6dmg_T1_50_un true
rm -rf ./transfer_params_75_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_T1/ 0 75 ./params_6dmg_T1 ./transfer_params_6dmg_T1_75_un true
rm -rf ./transfer_params_100_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_T1/ 0 100 ./params_6dmg_T1 ./transfer_params_6dmg_T1_100_un true
rm -rf ./transfer_params_5_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_T1/ 5 0 ./params_6dmg_T1 ./transfer_params_6dmg_T1_5_semi true
rm -rf ./transfer_params_10_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_T1/ 10 0 ./params_6dmg_T1 ./transfer_params_6dmg_T1_10_semi true
rm -rf ./transfer_params_15_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_T1/ 15 0 ./params_6dmg_T1 ./transfer_params_6dmg_T1_15_semi true
rm -rf ./transfer_params_20_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_T1/ 20 0 ./params_6dmg_T1 ./transfer_params_6dmg_T1_20_semi true
rm -rf ./transfer_params_25_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_T1/ 25 0 ./params_6dmg_T1 ./transfer_params_6dmg_T1_25_semi true
rm -rf ./transfer_params_30_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_T1/ 30 0 ./params_6dmg_T1 ./transfer_params_6dmg_T1_30_semi true
rm -rf ./transfer_params_100_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_T1/ 100 0 ./params_6dmg_T1 ./transfer_params_6dmg_T1_100_semi true

