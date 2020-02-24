import torch
# local class
import sys
sys.path.append("..")
import tools_6dmg
from cnnblstm import cnnblstm

TRAIN_PATH = r"../dataset/train_6dmg_C1"
TEST_PATH = r"../dataset/test_6dmg_C1"

if __name__ == '__main__':
	torch.manual_seed(2)

	# whether use cuda
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		cnnblstm = cnnblstm(params_file = "./params_6dmg_C1.pkl", use_cuda = use_cuda).cuda()
	else:
		cnnblstm = cnnblstm(params_file = "./params_6dmg_C1.pkl", use_cuda = use_cuda)
	print(cnnblstm)
	# get train_x, train_y
	X_all, y_all = tools_6dmg.preprocess(TRAIN_PATH)
	print(X_all.shape, y_all.shape)
	train_x = torch.from_numpy(X_all)
	train_y = torch.from_numpy(y_all)
	train_data = torch.utils.data.TensorDataset(train_x, train_y)
	# trainAllLayers
	cnnblstm.trainAllLayers(train_data)
	# get test_x, test_y
	X_all, y_all = tools_6dmg.preprocess(TEST_PATH)
	test_x = torch.from_numpy(X_all)
	test_y = torch.from_numpy(y_all)
	# get test accuracy
	cnnblstm.getTestAccuracy(test_x, test_y)

