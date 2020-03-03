#!/bin/sh
# unsuspected
rm -rf ./params_zh2zyt_uPCA
python train_cnnblstm_with_adabn.py train ../dataset/zhP30/ ./params_zh2zyt_uPCA false false
python train_cnnblstm_with_adabn.py test ../dataset/zytP30/ ./params_zh2zyt_uPCA false false
# semi-suspected
rm -rf ./transfer_params_zh2zyt_uPCA_100_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/zytP30/ 100 0 ./params_zh2zyt_uPCA ./transfer_params_zh2zyt_uPCA_100_semi false false false

