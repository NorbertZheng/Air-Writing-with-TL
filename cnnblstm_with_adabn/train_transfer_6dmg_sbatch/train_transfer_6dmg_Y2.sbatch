#!/bin/sh
rm -rf ./transfer_params_25_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_Y2/ 0 25 ./params_6dmg_Y2 ./transfer_params_6dmg_Y2_25_un true true
rm -rf ./transfer_params_50_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_Y2/ 0 50 ./params_6dmg_Y2 ./transfer_params_6dmg_Y2_50_un true true
rm -rf ./transfer_params_75_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_Y2/ 0 75 ./params_6dmg_Y2 ./transfer_params_6dmg_Y2_75_un true true
rm -rf ./transfer_params_100_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_Y2/ 0 100 ./params_6dmg_Y2 ./transfer_params_6dmg_Y2_100_un true true
rm -rf ./transfer_params_5_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_Y2/ 5 0 ./params_6dmg_Y2 ./transfer_params_6dmg_Y2_5_semi true true
rm -rf ./transfer_params_10_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_Y2/ 10 0 ./params_6dmg_Y2 ./transfer_params_6dmg_Y2_10_semi true true
rm -rf ./transfer_params_15_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_Y2/ 15 0 ./params_6dmg_Y2 ./transfer_params_6dmg_Y2_15_semi true true
rm -rf ./transfer_params_20_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_Y2/ 20 0 ./params_6dmg_Y2 ./transfer_params_6dmg_Y2_20_semi true true
rm -rf ./transfer_params_25_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_Y2/ 25 0 ./params_6dmg_Y2 ./transfer_params_6dmg_Y2_25_semi true true
rm -rf ./transfer_params_30_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_Y2/ 30 0 ./params_6dmg_Y2 ./transfer_params_6dmg_Y2_30_semi true true
rm -rf ./transfer_params_90_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_Y2/ 90 0 ./params_6dmg_Y2 ./transfer_params_6dmg_Y2_90_semi true true

