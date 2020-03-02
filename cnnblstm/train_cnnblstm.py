import torch
# local class
import sys
sys.path.append("..")
import tools
# from CNN_BLSTM import CNN_BLSTM
from cnnblstm import cnnblstm

# TRAIN_PATH = r"../dataset/train"
# TEST_PATH = r"../dataset/test"

"""
usage:
	python train_cnnblstm.py cmd path params_pkl enable_Kalman enable_PCA
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
		cnnblstm = cnnblstm(use_cuda = use_cuda, params_file = PARAMS_PATH_SYS).cuda()
	else:
		cnnblstm = cnnblstm(use_cuda = use_cuda, params_file = PARAMS_PATH_SYS)
	print(cnnblstm)
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
		cnnblstm.trainAllLayers(train_x, train_y)
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
		# get test accuracy
		cnnblstm.getTestAccuracy(test_x, test_y)
	else:
		print("CMD ERROR!")

