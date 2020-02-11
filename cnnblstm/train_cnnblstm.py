import torch
# local class
import tools
# from CNN_BLSTM import CNN_BLSTM
from cnnblstm import cnnblstm

TRAIN_PATH = r"../dataset/train"
TEST_PATH = r"../dataset/test"

if __name__ == '__main__':
	torch.manual_seed(2)

	# whether use cuda
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		cnnblstm = cnnblstm(use_cuda = 1).cuda()
	else:
		cnnblstm = cnnblstm(use_cuda = 0)
	print(cnnblstm)
	# get train_x, train_y
	Y, segments, maxlen_seg, n_files, seq_length = tools.getAllData(TRAIN_PATH)
	X_all, y_all, _ = tools.transferData(Y, segments, n_files, seq_length)
	train_x = torch.from_numpy(X_all)
	train_y = torch.from_numpy(y_all)
	train_data = torch.utils.data.TensorDataset(train_x, train_y)
	# trainAllLayers
	# cnnblstm.trainAllLayers(train_data)
	# get test_x, test_y
	Y, segments, maxlen_seg, n_files, seq_length = tools.getAllData(TRANSFER_TRAIN_PATH)
	X_all, y_all, _ = tools.transferData(Y, segments, n_files, seq_length)
	test_x = torch.from_numpy(X_all)
	test_y = torch.from_numpy(y_all)
	# get test accuracy
	cnnblstm.getTestAccuracy(test_x, test_y)

