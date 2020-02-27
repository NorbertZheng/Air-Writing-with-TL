#!/bin/sh
rm -rf ./transfer_params_25_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 0 25 ./transfer_params_25_un
rm -rf ./transfer_params_50_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 0 50 ./transfer_params_50_un
rm -rf ./transfer_params_75_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 0 75 ./transfer_params_75_un
rm -rf ./transfer_params_100_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 0 100 ./transfer_params_100_un
rm -rf ./transfer_params_5_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 5 0./transfer_params_5_semi
rm -rf ./transfer_params_10_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 10 0./transfer_params_10_semi
rm -rf ./transfer_params_15_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 15 0./transfer_params_15_semi
rm -rf ./transfer_params_20_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 20 0./transfer_params_20_semi
rm -rf ./transfer_params_25_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 25 0./transfer_params_25_semi
rm -rf ./transfer_params_30_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 30 0./transfer_params_30_semi
rm -rf ./transfer_params_100_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/transfer/ 100 0./transfer_params_100_semi
