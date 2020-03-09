#!/bin/sh
# unsuspected
rm -rf ./params_horizon2vertical_un_uPCA
python train_cnnblstm_with_adabn.py train ../dataset/horizon/ ./params_horizon2vertical_un_uPCA false false
python train_cnnblstm_with_adabn.py test ../dataset/vertical/ ./params_horizon2vertical_un_uPCA false false
# semi-suspected
rm -rf ./transfer_params_horizon2vertical_uPCA_100_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/vertical/ 100 0 ./params_horizon2vertical_un_uPCA ./transfer_params_horizon2vertical_uPCA_100_semi false false false
rm -rf ./transfer_params_horizon2vertical_uPCA_200_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/vertical/ 200 0 ./params_horizon2vertical_un_uPCA ./transfer_params_horizon2vertical_uPCA_200_semi false false false
rm -rf ./transfer_params_horizon2vertical_uPCA_500_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/vertical/ 500 0 ./params_horizon2vertical_un_uPCA ./transfer_params_horizon2vertical_uPCA_500_semi false false false
