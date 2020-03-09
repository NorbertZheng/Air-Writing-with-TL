#!/bin/sh
rm -rf ./params_6dmg_G3
python train_6dmg_cnnblstm_with_adabn.py ../dataset/6dmg_upper/G3/train ../dataset/6dmg_upper/G3/test ./params_6dmg_G3 false false
rm -rf ./transfer_params_6dmg_G3
python train_transfer_cnnblstm_with_adabn.py ../dataset/6dmg_upper/G3/test 50 0 ./params_6dmg_G3 ./transfer_params_6dmg_G3 false false true

