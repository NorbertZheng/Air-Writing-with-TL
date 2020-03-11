#!/bin/sh
# train naive cb
rm -rf ./params_rw_people_zh
python train_cnnblstm_with_adabn.py train ../dataset/rw_people/zh2zyt/train/ ./params_rw_people_zh.pkl false false
python train_cnnblstm_with_adabn.py test ../dataset/rw_people/zh2zyt/transfer/ ./params_rw_people_zh.pkl false false
