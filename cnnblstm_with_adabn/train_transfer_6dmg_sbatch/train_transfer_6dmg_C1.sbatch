#!/bin/sh
rm -rf ./transfer_params_25_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_C1/ 0 25 ./params_6dmg_C1 ./transfer_params_6dmg_C1_25_un true true
rm -rf ./transfer_params_50_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_C1/ 0 50 ./params_6dmg_C1 ./transfer_params_6dmg_C1_50_un true true
rm -rf ./transfer_params_75_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_C1/ 0 75 ./params_6dmg_C1 ./transfer_params_6dmg_C1_75_un true true
rm -rf ./transfer_params_100_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_C1/ 0 100 ./params_6dmg_C1 ./transfer_params_6dmg_C1_100_un true true
rm -rf ./transfer_params_5_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_C1/ 5 0 ./params_6dmg_C1 ./transfer_params_6dmg_C1_5_semi true true
rm -rf ./transfer_params_10_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_C1/ 10 0 ./params_6dmg_C1 ./transfer_params_6dmg_C1_10_semi true true
rm -rf ./transfer_params_15_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_C1/ 15 0 ./params_6dmg_C1 ./transfer_params_6dmg_C1_15_semi true true
rm -rf ./transfer_params_20_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_C1/ 20 0 ./params_6dmg_C1 ./transfer_params_6dmg_C1_20_semi true true
rm -rf ./transfer_params_25_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_C1/ 25 0 ./params_6dmg_C1 ./transfer_params_6dmg_C1_25_semi true true
rm -rf ./transfer_params_30_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_C1/ 30 0 ./params_6dmg_C1 ./transfer_params_6dmg_C1_30_semi true true
rm -rf ./transfer_params_90_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_C1/ 90 0 ./params_6dmg_C1 ./transfer_params_6dmg_C1_90_semi true true

