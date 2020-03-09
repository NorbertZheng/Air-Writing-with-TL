#!/bin/sh
rm -rf ./params_6dmg_Y1
python train_6dmg_cnnblstm_with_adabn.py ../dataset/6dmg_upper/Y1/train ../dataset/6dmg_upper/Y1/test ./params_6dmg_Y1 false false
rm -rf ./transfer_params_6dmg_Y1
python train_transfer_cnnblstm_with_adabn.py ../dataset/6dmg_upper/Y1/test 100 0 ./params_6dmg_Y1 ./transfer_params_6dmg_Y1 false false true

