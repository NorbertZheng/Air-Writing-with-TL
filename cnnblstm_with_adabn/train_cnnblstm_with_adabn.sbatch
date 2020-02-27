#!/bin/sh
rm -rf ./params_horizon2vertical_un
python train_cnnblstm_with_adabn.py train ../dataset/horizon/ ./params_horizon2vertical_un
python train_cnnblstm_with_adabn.py test ../dataset/vertical ./params_horizon2vertical_un
rm -rf ./params_horizon2vertical_un
