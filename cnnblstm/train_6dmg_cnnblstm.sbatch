#!/bin/sh
rm -rf ./params_6dmg.pkl
python train_6dmg_cnnblstm.py train ../dataset/train_6dmg_C1 true
python train_6dmg_cnnblstm.py test ../dataset/test_6dmg_C1 true
rm -rf ./params_6dmg.pkl
python train_6dmg_cnnblstm.py train ../dataset/train_6dmg_J1 true
python train_6dmg_cnnblstm.py test ../dataset/test_6dmg_J1 true
rm -rf ./params_6dmg.pkl
python train_6dmg_cnnblstm.py train ../dataset/train_6dmg_M1 true
python train_6dmg_cnnblstm.py test ../dataset/test_6dmg_M1 true
rm -rf ./params_6dmg.pkl
python train_6dmg_cnnblstm.py train ../dataset/train_6dmg_T1 true
python train_6dmg_cnnblstm.py test ../dataset/test_6dmg_T1 true
rm -rf ./params_6dmg.pkl
python train_6dmg_cnnblstm.py train ../dataset/train_6dmg_Y1 true
python train_6dmg_cnnblstm.py test ../dataset/test_6dmg_Y1 true
rm -rf ./params_6dmg.pkl
python train_6dmg_cnnblstm.py train ../dataset/train_6dmg_Y2 true
python train_6dmg_cnnblstm.py test ../dataset/test_6dmg_Y2 true
rm -rf ./params_6dmg.pkl
