#!/bin/sh
rm -rf ./transfer_params_25_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 0 25 ./params ./transfer_params_25_un true true false
rm -rf ./transfer_params_50_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 0 50 ./params ./transfer_params_50_un true true false
rm -rf ./transfer_params_75_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 0 75 ./params ./transfer_params_75_un true true false
rm -rf ./transfer_params_100_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 0 100 ./params ./transfer_params_100_un true true false
rm -rf ./transfer_params_5_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 5 0 ./params ./transfer_params_5_semi true true false
rm -rf ./transfer_params_10_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 10 0 ./params ./transfer_params_10_semi true true false
rm -rf ./transfer_params_15_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 15 0 ./params ./transfer_params_15_semi true true false
rm -rf ./transfer_params_20_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 20 0 ./params ./transfer_params_20_semi true true false
rm -rf ./transfer_params_25_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 25 0 ./params ./transfer_params_25_semi true true false
rm -rf ./transfer_params_30_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 30 0 ./params ./transfer_params_30_semi true true false
rm -rf ./transfer_params_100_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 100 0 ./params ./transfer_params_100_semi true true false

