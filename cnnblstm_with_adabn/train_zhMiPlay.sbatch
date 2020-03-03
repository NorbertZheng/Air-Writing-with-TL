#!/bin/sh
rm -rf ./params_zhMiPlay
python train_cnnblstm_with_adabn.py train ../dataset/zhMiPlay_train ./params_zhMiPlay false false
python train_cnnblstm_with_adabn.py test ../dataset/zhMiPlay_test ./params_zhMiPlay false false

