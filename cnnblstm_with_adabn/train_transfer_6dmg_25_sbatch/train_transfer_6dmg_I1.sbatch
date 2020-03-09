#!/bin/sh
rm -rf ./params_6dmg_I1
python train_6dmg_cnnblstm_with_adabn.py ../dataset/6dmg_upper/I1/train ../dataset/6dmg_upper/I1/test ./params_6dmg_I1 false false
rm -rf ./transfer_params_6dmg_I1
python train_transfer_cnnblstm_with_adabn.py ../dataset/6dmg_upper/I1/test 100 0 ./params_6dmg_I1 ./transfer_params_6dmg_I1 false false true

