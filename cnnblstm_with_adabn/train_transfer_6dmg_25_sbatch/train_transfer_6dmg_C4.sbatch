#!/bin/sh
rm -rf ./params_6dmg_C4
python train_6dmg_cnnblstm_with_adabn.py ../dataset/6dmg_upper/C4/train ../dataset/6dmg_upper/C4/test ./params_6dmg_C4 false false
rm -rf ./transfer_params_6dmg_C4
python train_transfer_cnnblstm_with_adabn.py ../dataset/6dmg_upper/C4/test 50 0 ./params_6dmg_C4 ./transfer_params_6dmg_C4 false false true

