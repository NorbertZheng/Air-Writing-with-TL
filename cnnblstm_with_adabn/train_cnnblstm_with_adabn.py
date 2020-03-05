import torch
import numpy as np
# local class
import sys
sys.path.append("..")
sys.path.append("../deepCoral")
import tools
from utils import AverageMeter
from cnnblstm_with_adabn import cnnblstm_with_adabn

# enable_PCA = True
# TRAIN_PATH = r"../dataset/train"
# TEST_PATH = r"../dataset/test"
N_SET = 500

"""
usage:
	python train_cnnblstm_with_adabn.py cmd path params_dir enable_Kalman enable_PCA
"""

if __name__ == '__main__':
	torch.manual_seed(2)

	# get TRAIN_PATH_SYS & TEST_PATH_SYS & PARAMS_PATH_SYS
	CMD = sys.argv[1]
	PATH_SYS = sys.argv[2]
	PARAMS_PATH_SYS = sys.argv[3]
	enable_Kalman = (sys.argv[4] == "true")
	enable_PCA = (sys.argv[5] == "true")
	# whether use cuda
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		m_cnnblstm_with_adabn = cnnblstm_with_adabn(use_cuda = use_cuda, params_dir = PARAMS_PATH_SYS).cuda()
	else:
		m_cnnblstm_with_adabn = cnnblstm_with_adabn(use_cuda = use_cuda, params_dir = PARAMS_PATH_SYS)
	print(m_cnnblstm_with_adabn)
	if (CMD == "train"):
		# get train_x, train_y
		Y, segments, maxlen_seg, n_files, seq_length = tools.getAllData(PATH_SYS)
		train_x, train_y, _ = tools.transferData(Y, segments, n_files, seq_length)
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
		m_cnnblstm_with_adabn.trainAllLayers(train_x, train_y, n_epoches = 20)
	elif (CMD == "test"):
		# get test_x, test_y
		Y, segments, maxlen_seg, n_files, seq_length = tools.getAllData(PATH_SYS)
		test_x, test_y, _ = tools.transferData(Y, segments, n_files, seq_length)
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
		avc = AverageMeter().reset()
		for i in range(test_x.shape[0] // N_SET):
			test_x_batch = test_x[(i * N_SET):((i + 1) * N_SET), :, :]
			test_y_batch = test_y[(i * N_SET):((i + 1) * N_SET)]
			# get test accuracy
			acc_batch = m_cnnblstm_with_adabn.getTestAccuracy(test_x_batch, test_y_batch)
			avc.update(acc_batch, n = N_SET)
		if test_x.shape[0] % N_SET == 0:
			print("Accuracy: ", str(avc.avg))
		else:
			test_x_batch = test_x[-(test_x.shape[0] % N_SET):, :, :]
			test_y_batch = test_y[-(test_x.shape[0] % N_SET):]
			# get test accuracy
			acc_batch = m_cnnblstm_with_adabn.getTestAccuracy(test_x_batch, test_y_batch)
			avc.update(acc_batch, n = (test_x.shape[0] % N_SET))
			print("Accuracy: ", str(avc.avg))
	else:
		print("CMD ERROR!")

