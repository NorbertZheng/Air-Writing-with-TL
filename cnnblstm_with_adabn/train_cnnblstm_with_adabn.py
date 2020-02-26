import torch
import numpy as np
# local class
import sys
sys.path.append("..")
import tools
from cnnblstm_with_adabn import cnnblstm_with_adabn

enable_PCA = True
# TRAIN_PATH = r"../dataset/train"
# TEST_PATH = r"../dataset/test"

if __name__ == '__main__':
	torch.manual_seed(2)

	# whether use cuda
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		m_cnnblstm_with_adabn = cnnblstm_with_adabn(use_cuda = 1).cuda()
	else:
		m_cnnblstm_with_adabn = cnnblstm_with_adabn(use_cuda = 0)
	print(m_cnnblstm_with_adabn)
	# get TRAIN_PATH_SYS & TEST_PATH_SYS
	TRAIN_PATH_SYS = sys.argv[1]
	TEST_PATH_SYS = sys.argv[2]
	# get train_x, train_y
	Y, segments, maxlen_seg, n_files, seq_length = tools.getAllData(TRAIN_PATH_SYS)
	train_x, train_y, _ = tools.transferData(Y, segments, n_files, seq_length)
	# enable PCA
	if enable_PCA:
		train_x = tools.PCA_Xs(train_x).astype(np.float32)
		print(train_x.shape)
	train_x = torch.from_numpy(train_x)
	train_y = torch.from_numpy(train_y)
	# trainAllLayers
	m_cnnblstm_with_adabn.trainAllLayers(train_x, train_y)
	# get test_x, test_y
	Y, segments, maxlen_seg, n_files, seq_length = tools.getAllData(TEST_PATH_SYS)
	test_x, test_y, _ = tools.transferData(Y, segments, n_files, seq_length)
	# enable PCA
	if enable_PCA:
		test_x = tools.PCA_Xs(test_x).astype(np.float32)
		print(test_x.shape)
	test_x = torch.from_numpy(test_x)
	test_y = torch.from_numpy(test_y)
	# get test accuracy
	m_cnnblstm_with_adabn.getTestAccuracy(test_x, test_y)

