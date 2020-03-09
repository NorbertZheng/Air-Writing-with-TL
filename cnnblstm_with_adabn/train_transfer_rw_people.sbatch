#!/bin/sh
# train naive cba
rm -rf ./params_rw_people_zh
python train_cnnblstm_with_adabn.py train ../dataset/rw_people/zh2zyt/train/ ./params_rw_people_zh false false
python train_cnnblstm_with_adabn.py test ../dataset/rw_people/zh2zyt/transfer/ ./params_rw_people_zh false false
# unsuspected
rm -rf ./params_transfer_rw_people_zh_100_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_people/zh2zyt/transfer/ 0 100 ./params_rw_people_zh ./params_transfer_rw_people_zh_100_un false false false
# semi-suspected
rm -rf ./params_transfer_rw_people_zh_100_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_people/zh2zyt/transfer/ 100 0 ./params_rw_people_zh ./params_transfer_rw_people_zh_100_semi false false false

