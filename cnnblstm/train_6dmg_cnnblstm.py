import torch
import numpy as np
# local class
import sys
sys.path.append("..")
import tools
import tools_6dmg
from cnnblstm import cnnblstm

TRAIN_PATH = r"../dataset/train_6dmg"
TEST_PATH = r"../dataset/test_6dmg"

"""
usage:
	python train_6dmg_cnnblstm.py cmd path enable_Kalman enable_PCA
"""

if __name__ == '__main__':
	torch.manual_seed(2)

	# get TRAIN_PATH_SYS & TEST_PATH_SYS
	CMD = sys.argv[1]
	PATH_SYS = sys.argv[2]
	enable_Kalman = (sys.argv[3] == "true")
	enable_PCA = (sys.argv[4] == "true")
	# whether use cuda
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		cnnblstm = cnnblstm(params_file = "./params_6dmg.pkl", use_cuda = use_cuda).cuda()
	else:
		cnnblstm = cnnblstm(params_file = "./params_6dmg.pkl", use_cuda = use_cuda)
	print(cnnblstm)
	if (CMD == "train"):
		# get train_x, train_y
		train_x, train_y = tools_6dmg.preprocess(PATH_SYS)
		# enable Kalman
		if enable_Kalman:
			train_x = tools.Kalman_Xs(train_x).astype(np.float32)
			print(train_x.shape)
		# enable PCA
		if enable_PCA:
			train_x = tools.PCA_Xs(train_x).astype(np.float32)
			print(train_x.shape)
		train_x = torch.from_numpy(train_x)
		train_y = torch.from_numpy(train_y)
		# trainAllLayers
		cnnblstm.trainAllLayers(train_x, train_y, n_epoches = 30)
	elif (CMD == "test"):
		# get test_x, test_y
		test_x, test_y = tools_6dmg.preprocess(PATH_SYS)
		# enable Kalman
		if enable_Kalman:
			test_x = tools.Kalman_Xs(test_x).astype(np.float32)
			print(test_x.shape)
		# enable PCA
		if enable_PCA:
			test_x = tools.PCA_Xs(test_x).astype(np.float32)
			print(test_x.shape)
		test_x = torch.from_numpy(test_x)
		test_y = torch.from_numpy(test_y)
		# get test accuracy
		cnnblstm.getTestAccuracy(test_x, test_y)
	else:
		print("CMD ERROR!")

