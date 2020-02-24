import torch
# local class
import sys
sys.path.append("..")
import tools_6dmg
sys.path.append("../network")
import Coral
from cnnblstm_with_adabn import cnnblstm_with_adabn

enable_CORAL = True
TRAIN_PATH = r"../dataset/train_6dmg"
TEST_PATH = r"../dataset/test_6dmg"

if __name__ == '__main__':
	torch.manual_seed(2)

	# whether use cuda
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		m_cnnblstm_with_adabn = cnnblstm_with_adabn(use_cuda = 1, params_dir = "./params_6dmg", enable_CORAL = enable_CORAL).cuda()
	else:
		m_cnnblstm_with_adabn = cnnblstm_with_adabn(use_cuda = 0, params_dir = "./params_6dmg", enable_CORAL = enable_CORAL)
	print(m_cnnblstm_with_adabn)
	# get train_x, train_y
	X_all, y_all = tools_6dmg.preprocess(TRAIN_PATH)
	# get test_x, test_y
	X_all, y_all = tools_6dmg.preprocess(TEST_PATH)
	# if enable_CORAL
	if enable_CORAL:
		# record old size
		old_train_x_size = train_x.shape
		assert len(old_train_x_size) == 3
		old_test_x_size = test_x.shape
		assert len(old_test_x_size) == 3
		# resize train_x & test_x
		train_x = train_x.resize((old_train_x_size[0], old_train_x_size[1] * old_train_x_size[2]))
		test_x = test_x.resize((old_test_x_size[0], old_test_x_size[1] * old_test_x_size[2]))
		# get 
		train_x = Coral.CORAL_np(train_x, test_x)
		# resize train_x & test_x
		train_x = train_x.resize((old_train_x_size[0], old_train_x_size[1], old_train_x_size[2]))
		test_x = test_x.resize((old_test_x_size[0], old_test_x_size[1], old_test_x_size[2]))
	# init as tensor
	if use_cuda:
		train_x = torch.from_numpy(X_all).cuda()
		train_y = torch.from_numpy(y_all).cuda()
	else:
		train_x = torch.from_numpy(X_all)
		train_y = torch.from_numpy(y_all)
	# init as tensor
	if use_cuda:
		test_x = torch.from_numpy(X_all).cuda()
		test_y = torch.from_numpy(y_all).cuda()
	else:
		test_x = torch.from_numpy(X_all)
		test_y = torch.from_numpy(y_all)
	print(train_x.size(), test_x.size())
	# trainAllLayers
	m_cnnblstm_with_adabn.trainAllLayers(train_x, train_y, test_x)
	# get test accuracy
	m_cnnblstm_with_adabn.getTestAccuracy(test_x, test_y)

