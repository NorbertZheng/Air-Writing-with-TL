#!/bin/sh
# train naive cba
rm -rf ./params_rw_pose_h
python train_cnnblstm_with_adabn.py train ../dataset/rw_pose/h2v/train/ ./params_rw_pose_h false false
python train_cnnblstm_with_adabn.py test ../dataset/rw_pose/h2v/transfer/ ./params_rw_pose_h false false
# unsuspected
rm -rf ./params_transfer_rw_pose_h_100_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_pose/h2v/transfer/ 0 100 ./params_rw_pose_h ./params_transfer_rw_pose_h_100_un false false false
# semi-suspected
rm -rf ./params_transfer_rw_pose_h_30_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_pose/h2v/transfer/ 30 0 ./params_rw_pose_h ./params_transfer_rw_pose_h_30_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_pose_h_50_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_pose/h2v/transfer/ 50 0 ./params_rw_pose_h ./params_transfer_rw_pose_h_50_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_pose_h_100_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_pose/h2v/transfer/ 100 0 ./params_rw_pose_h ./params_transfer_rw_pose_h_100_semi false false false

