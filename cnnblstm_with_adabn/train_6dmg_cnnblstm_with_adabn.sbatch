#!/bin/sh
rm -rf ./params_6dmg
python train_6dmg_cnnblstm_with_adabn.py ../dataset/train_6dmg_C1 ../dataset/test_6dmg_C1
rm -rf ./params_6dmg
python train_6dmg_cnnblstm_with_adabn.py ../dataset/train_6dmg_J1 ../dataset/test_6dmg_J1
rm -rf ./params_6dmg
python train_6dmg_cnnblstm_with_adabn.py ../dataset/train_6dmg_M1 ../dataset/test_6dmg_M1
rm -rf ./params_6dmg
python train_6dmg_cnnblstm_with_adabn.py ../dataset/train_6dmg_T1 ../dataset/test_6dmg_T1
rm -rf ./params_6dmg
python train_6dmg_cnnblstm_with_adabn.py ../dataset/train_6dmg_Y1 ../dataset/test_6dmg_Y1
rm -rf ./params_6dmg
python train_6dmg_cnnblstm_with_adabn.py ../dataset/train_6dmg_Y2 ../dataset/test_6dmg_Y2
rm -rf ./params_6dmg

