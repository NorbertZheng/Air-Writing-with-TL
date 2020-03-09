#!/bin/sh
rm -rf ./params_6dmg_I3
python train_6dmg_cnnblstm_with_adabn.py ../dataset/6dmg_upper/I3/train ../dataset/6dmg_upper/I3/test ./params_6dmg_I3 false false
rm -rf ./transfer_params_6dmg_I3
python train_transfer_cnnblstm_with_adabn.py ../dataset/6dmg_upper/I3/test 100 0 ./params_6dmg_I3 ./transfer_params_6dmg_I3 false false true

