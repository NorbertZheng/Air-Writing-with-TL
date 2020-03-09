#!/bin/sh
rm -rf ./params_6dmg_I2
python train_6dmg_cnnblstm_with_adabn.py ../dataset/6dmg_upper/I2/train ../dataset/6dmg_upper/I2/test ./params_6dmg_I2 false false
rm -rf ./transfer_params_6dmg_I2
python train_transfer_cnnblstm_with_adabn.py ../dataset/6dmg_upper/I2/test 50 0 ./params_6dmg_I2 ./transfer_params_6dmg_I2 false false true

