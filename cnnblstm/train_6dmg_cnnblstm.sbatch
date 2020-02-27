#!/bin/sh
rm -rf ./params_6dmg.pkl
python train_6dmg_cnnblstm.py train ../dataset/train_6dmg_C1
python train_6dmg_cnnblstm.py test ../dataset/test_6dmg_C1
rm -rf ./params_6dmg.pkl
python train_6dmg_cnnblstm.py train ../dataset/train_6dmg_J1
python train_6dmg_cnnblstm.py test ../dataset/test_6dmg_J1
rm -rf ./params_6dmg.pkl
python train_6dmg_cnnblstm.py train ../dataset/train_6dmg_M1
python train_6dmg_cnnblstm.py test ../dataset/test_6dmg_M1
rm -rf ./params_6dmg.pkl
python train_6dmg_cnnblstm.py train ../dataset/train_6dmg_T1
python train_6dmg_cnnblstm.py test ../dataset/test_6dmg_T1
rm -rf ./params_6dmg.pkl
python train_6dmg_cnnblstm.py train ../dataset/train_6dmg_Y1
python train_6dmg_cnnblstm.py test ../dataset/test_6dmg_Y1
rm -rf ./params_6dmg.pkl
python train_6dmg_cnnblstm.py train ../dataset/train_6dmg_Y2
python train_6dmg_cnnblstm.py test ../dataset/test_6dmg_Y2
rm -rf ./params_6dmg.pkl
