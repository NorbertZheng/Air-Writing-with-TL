#!/bin/sh
# unsuspected
# rm -rf ./params_horizon2vertical_un_PCA
# python train_cnnblstm_with_adabn.py train ../dataset/horizon/ ./params_horizon2vertical_un_PCA true true
# python train_cnnblstm_with_adabn.py test ../dataset/vertical/ ./params_horizon2vertical_un_PCA true true
# semi-suspected
rm -rf ./transfer_params_horizon2vertical_100_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/vertical/ 100 0 ./params_horizon2vertical_un_PCA ./transfer_params_horizon2vertical_100_semi true true false
rm -rf ./transfer_params_horizon2vertical_200_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/vertical/ 200 0 ./params_horizon2vertical_un_PCA ./transfer_params_horizon2vertical_200_semi true true false
rm -rf ./transfer_params_horizon2vertical_500_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/vertical/ 500 0 ./params_horizon2vertical_un_PCA ./transfer_params_horizon2vertical_500_semi true true false

