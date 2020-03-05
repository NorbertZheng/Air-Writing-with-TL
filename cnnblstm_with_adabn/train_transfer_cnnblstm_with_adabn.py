import torch
import numpy as np
# local class
import sys
sys.path.append("..")
sys.path.append("../deepCoral")
import tools
import tools_6dmg
from utils import AverageMeter
from transfer_cnnblstm_with_adabn import transfer_cnnblstm_with_adabn

# N_TRAINSET = 0
# TRANSFER_PATH = r"../dataset/transfer"

"""
usage:
	python train_transfer_cnnblstm_with_adabn.py transfer_path n_trainset n_testset params_dir transfer_params_dir enable_PCA enable_Kalman is_6dmg
"""

if __name__ == '__main__':
	torch.manual_seed(2)

	# get TRANSFER_PATH & 
	TRANSFER_PATH = sys.argv[1]
	N_TRAINSET = int(sys.argv[2])
	N_TESTSET = int(sys.argv[3])
	PARAMS_PATH = sys.argv[4]
	TRANSFER_PARAMS_PATH = sys.argv[5]
	enable_PCA = (sys.argv[6] == "true")
	enable_Kalman = (sys.argv[7] == "true")
	IS_6DMG = (sys.argv[8] == "true")
	# whether use cuda
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		m_transfer_cnnblstm_with_adabn = transfer_cnnblstm_with_adabn(use_cuda = use_cuda, params_dir = PARAMS_PATH, transfer_params_dir = TRANSFER_PARAMS_PATH).cuda()
	else:
		print(use_cuda)
		m_transfer_cnnblstm_with_adabn = transfer_cnnblstm_with_adabn(use_cuda = use_cuda, params_dir = PARAMS_PATH, transfer_params_dir = TRANSFER_PARAMS_PATH)
	print(m_transfer_cnnblstm_with_adabn)
	# get transfer_x & transfer_y
	if IS_6DMG:
		transfer_x, transfer_y = tools_6dmg.preprocess(TRANSFER_PATH)
	else:
		Y, segments, maxlen_seg, n_files, seq_length = tools.getAllData(TRANSFER_PATH)
		transfer_x, transfer_y, _ = tools.transferData(Y, segments, n_files, seq_length)
	# get permutation index
	per = np.random.permutation(transfer_x.shape[0])
	# premute transfer_x & transfer_y
	transfer_x = transfer_x[per, :, :]
	transfer_y = transfer_y[per]
	# enable Kalman
	if enable_Kalman:
		transfer_x = tools.Kalman_Xs(transfer_x).astype(np.float32)
		print(transfer_x.shape)
	# enable PCA
	if enable_PCA:
		transfer_x = tools.PCA_Xs(transfer_x).astype(np.float32)
		print(transfer_x.shape)
	# get train_x & train_y
	if N_TRAINSET != 0:
		train_x = torch.from_numpy(transfer_x[:N_TRAINSET, :, :])
		train_y = torch.from_numpy(transfer_y[:N_TRAINSET])
		print(train_x.shape, train_y.shape)
	else:
		train_x, train_y = None, None
	# get test_x & test_y
	if (N_TRAINSET != 0):
		test_x = torch.from_numpy(transfer_x[N_TRAINSET:, :, :])
		test_y = torch.from_numpy(transfer_y[N_TRAINSET:])
		print(test_x.shape, test_y.shape)
	elif (N_TESTSET != 0):
		test_x = torch.from_numpy(transfer_x[:N_TESTSET, :, :])
		test_y = torch.from_numpy(transfer_y[:N_TESTSET])
		print(test_x.shape, test_y.shape)
	else:
		print("N_TRAINSET or N_TESTSET ERROR!")
	# get test acc
	avc = AverageMeter()
	avc.reset()
	for i in range(test_x.shape[0] // N_SET):
		test_x_batch = test_x[(i * N_SET):((i + 1) * N_SET), :, :]
		test_y_batch = test_y[(i * N_SET):((i + 1) * N_SET)]
		# get test accuracy
		acc_batch = m_transfer_cnnblstm_with_adabn.getTestAccuracy(test_x_batch, test_y_batch)
		avc.update(acc_batch, n = N_SET)
	if test_x.shape[0] % N_SET == 0:
		print("Accuracy: ", str(avc.avg))
	else:
		test_x_batch = test_x[-(test_x.shape[0] % N_SET):, :, :]
		test_y_batch = test_y[-(test_x.shape[0] % N_SET):]
		# get test accuracy
		acc_batch = m_transfer_cnnblstm_with_adabn.getTestAccuracy(test_x_batch, test_y_batch)
		avc.update(acc_batch, n = (test_x.shape[0] % N_SET))
		print("Accuracy: ", str(avc.avg))
	# trainAllLayers
	m_transfer_cnnblstm_with_adabn.trainAllLayers(train_x, train_y, test_x = test_x, n_epoches = 20)
	# get test acc
	avc = AverageMeter()
	avc.reset()
	for i in range(test_x.shape[0] // N_SET):
		test_x_batch = test_x[(i * N_SET):((i + 1) * N_SET), :, :]
		test_y_batch = test_y[(i * N_SET):((i + 1) * N_SET)]
		# get test accuracy
		acc_batch = m_transfer_cnnblstm_with_adabn.getTestAccuracy(test_x_batch, test_y_batch)
		avc.update(acc_batch, n = N_SET)
	if test_x.shape[0] % N_SET == 0:
		print("Accuracy: ", str(avc.avg))
	else:
		test_x_batch = test_x[-(test_x.shape[0] % N_SET):, :, :]
		test_y_batch = test_y[-(test_x.shape[0] % N_SET):]
		# get test accuracy
		acc_batch = m_transfer_cnnblstm_with_adabn.getTestAccuracy(test_x_batch, test_y_batch)
		avc.update(acc_batch, n = (test_x.shape[0] % N_SET))
		print("Accuracy: ", str(avc.avg))

