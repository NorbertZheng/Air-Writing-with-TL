#!/bin/sh
# rm -rf ./params_dong2zh.pkl
python train_cnnblstm.py train ../dataset/train/ ./params_dong2zh.pkl
python train_cnnblstm.py test ../dataset/transfer/ ./params_dong2zh.pkl
# rm -rf ./params_dong2zh.pkl
