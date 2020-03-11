#!/bin/sh
# train naive cb
rm -rf ./params_rw_pose_h
python train_cnnblstm_with_adabn.py train ../dataset/rw_pose/h2v/train/ ./params_rw_pose_h.pkl false false
python train_cnnblstm_with_adabn.py test ../dataset/rw_pose/h2v/transfer/ ./params_rw_pose_h.pkl false false
