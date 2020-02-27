#!/bin/sh
# rm -rf ./params_horizon2vertical_un_PCA
python train_cnnblstm_with_adabn.py train ../dataset/horizon/ ./params_horizon2vertical_un_PCA true
python train_cnnblstm_with_adabn.py test ../dataset/vertical ./params_horizon2vertical_un_PCA true
# rm -rf ./params_horizon2vertical_un_PCA
# rm -rf ./params_dong2zh_un_nPCA
python train_cnnblstm_with_adabn.py train ../dataset/train/ ./params_dong2zh_un_nPCA false
python train_cnnblstm_with_adabn.py test ../dataset/transfer/ ./params_dong2zh_un_nPCA false
# rm -rf ./params_dong2zh_un_nPCA
