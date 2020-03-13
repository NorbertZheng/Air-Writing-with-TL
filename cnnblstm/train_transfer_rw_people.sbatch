#!/bin/sh
module load nvidia/cuda/10.0
# train naive cb
rm -rf ./params_rw_people_zh.pkl
python train_cnnblstm.py train ../dataset/rw_people/zh2zyt/train/ ./params_rw_people_zh.pkl false false
python train_cnnblstm.py test ../dataset/rw_people/zh2zyt/transfer/ ./params_rw_people_zh.pkl false false
