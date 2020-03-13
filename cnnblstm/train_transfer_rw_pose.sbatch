#!/bin/sh
module load nvidia/cuda/10.0
# train naive cb
rm -rf ./params_rw_pose_h.pkl
python train_cnnblstm.py train ../dataset/rw_pose/h2v/train/ ./params_rw_pose_h.pkl false false
python train_cnnblstm.py test ../dataset/rw_pose/h2v/transfer/ ./params_rw_pose_h.pkl false false
