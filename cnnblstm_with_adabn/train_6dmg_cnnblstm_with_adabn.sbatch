#!/bin/sh
rm -rf ./params_6dmg_C1
python train_6dmg_cnnblstm_with_adabn.py ../dataset/train_6dmg_C1 ../dataset/test_6dmg_C1 ./params_6dmg_C1 true
rm -rf ./params_6dmg_J1
python train_6dmg_cnnblstm_with_adabn.py ../dataset/train_6dmg_J1 ../dataset/test_6dmg_J1 ./params_6dmg_J1 true
rm -rf ./params_6dmg_M1
python train_6dmg_cnnblstm_with_adabn.py ../dataset/train_6dmg_M1 ../dataset/test_6dmg_M1 ./params_6dmg_M1 true
rm -rf ./params_6dmg_T1
python train_6dmg_cnnblstm_with_adabn.py ../dataset/train_6dmg_T1 ../dataset/test_6dmg_T1 ./params_6dmg_T1 true
rm -rf ./params_6dmg_Y1
python train_6dmg_cnnblstm_with_adabn.py ../dataset/train_6dmg_Y1 ../dataset/test_6dmg_Y1 ./params_6dmg_Y1 true
rm -rf ./params_6dmg_Y2
python train_6dmg_cnnblstm_with_adabn.py ../dataset/train_6dmg_Y2 ../dataset/test_6dmg_Y2 ./params_6dmg_Y2 true

