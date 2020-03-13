#!/bin/sh
module load nvidia/cuda/10.0
# train naive cba
rm -rf ./params_rw_pose_h
python train_cnnblstm_with_adabn.py train ../dataset/rw_pose/h2v/train/ ./params_rw_pose_h false false
python train_cnnblstm_with_adabn.py test ../dataset/rw_pose/h2v/transfer/ ./params_rw_pose_h false false
# unsuspected
rm -rf ./params_transfer_rw_pose_h_100_un
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_pose/h2v/transfer/ 0 100 ./params_rw_pose_h ./params_transfer_rw_pose_h_100_un false false false
# semi-suspected
rm -rf ./params_transfer_rw_pose_h_10_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_pose/h2v/transfer/ 10 0 ./params_rw_pose_h ./params_transfer_rw_pose_h_10_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_pose_h_20_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_pose/h2v/transfer/ 20 0 ./params_rw_pose_h ./params_transfer_rw_pose_h_20_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_pose_h_30_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_pose/h2v/transfer/ 30 0 ./params_rw_pose_h ./params_transfer_rw_pose_h_30_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_pose_h_40_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_pose/h2v/transfer/ 40 0 ./params_rw_pose_h ./params_transfer_rw_pose_h_40_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_pose_h_50_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_pose/h2v/transfer/ 50 0 ./params_rw_pose_h ./params_transfer_rw_pose_h_50_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_pose_h_60_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_pose/h2v/transfer/ 60 0 ./params_rw_pose_h ./params_transfer_rw_pose_h_60_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_pose_h_70_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_pose/h2v/transfer/ 70 0 ./params_rw_pose_h ./params_transfer_rw_pose_h_70_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_pose_h_80_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_pose/h2v/transfer/ 80 0 ./params_rw_pose_h ./params_transfer_rw_pose_h_80_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_pose_h_90_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_pose/h2v/transfer/ 90 0 ./params_rw_pose_h ./params_transfer_rw_pose_h_90_semi false false false
# semi-suspected
rm -rf ./params_transfer_rw_pose_h_100_semi
python train_transfer_cnnblstm_with_adabn.py ../dataset/rw_pose/h2v/transfer/ 100 0 ./params_rw_pose_h ./params_transfer_rw_pose_h_100_semi false false false

