import torch
# local class
import sys
sys.path.append("..")
import tools
# from CNN_BLSTM import CNN_BLSTM
from cnnblstm import cnnblstm

# TRAIN_PATH = r"../dataset/train"
# TEST_PATH = r"../dataset/test"

if __name__ == '__main__':
	torch.manual_seed(2)

	# get TRAIN_PATH_SYS & TEST_PATH_SYS & PARAMS_PATH_SYS
	CMD = sys.argv[1]
	PATH_SYS = sys.argv[2]
	PARAMS_PATH_SYS = sys.argv[3]
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
		X_all, y_all, _ = tools.transferData(Y, segments, n_files, seq_length)
		train_x = torch.from_numpy(X_all)
		train_y = torch.from_numpy(y_all)
		# trainAllLayers
		cnnblstm.trainAllLayers(train_x, train_y)
	elif (CMD == "test"):
		# get test_x, test_y
		Y, segments, maxlen_seg, n_files, seq_length = tools.getAllData(PATH_SYS)
		X_all, y_all, _ = tools.transferData(Y, segments, n_files, seq_length)
		test_x = torch.from_numpy(X_all)
		test_y = torch.from_numpy(y_all)
		# get test accuracy
		cnnblstm.getTestAccuracy(test_x, test_y)
	else:
		print("CMD ERROR!")

