#!/bin/sh
rm -rf ./params_6dmg_Y2
python train_6dmg_cnnblstm_with_adabn.py ../dataset/6dmg_upper/Y2/train ../dataset/6dmg_upper/Y2/test ./params_6dmg_Y2 false false
rm -rf ./transfer_params_6dmg_Y2
python train_transfer_cnnblstm_with_adabn.py ../dataset/6dmg_upper/Y2/test 50 0 ./params_6dmg_Y2 ./transfer_params_6dmg_Y2 false false true

