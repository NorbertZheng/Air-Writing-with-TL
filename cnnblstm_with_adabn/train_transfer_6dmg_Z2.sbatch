#!/bin/sh
rm -rf ./params_6dmg_Z2
python train_6dmg_cnnblstm_with_adabn.py ../dataset/train_6dmg_Z2 ../dataset/test_6dmg_Z2 /params_6dmg_Z2 false false
rm -rf ./transfer_params_6dmg_Z2
python train_transfer_cnnblstm_with_adabn.py ../dataset/test_6dmg_Z2 100 0 ./params_6dmg_Z2 ./transfer_params_6dmg_Z2 false false true

