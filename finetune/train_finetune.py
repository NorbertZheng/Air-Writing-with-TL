import torch
# local class
import sys
sys.path.append("..")
import tools
from finetune import finetune

TRAIN_PATH = r"../dataset/transfer_train"
TEST_PATH = r"../dataset/transfer_test"
# TEST_PATH = r"./dataset/test"

if __name__ == '__main__':
	torch.manual_seed(2)

	# whether use cuda
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		m_finetune = finetune(use_cuda = 1).cuda()
	else:
		m_finetune = finetune(use_cuda = 0)
	print(m_finetune)
	# get test_x, test_y
	Y, segments, maxlen_seg, n_files, seq_length = tools.getAllData(TEST_PATH)
	X_all, y_all, _ = tools.transferData(Y, segments, n_files, seq_length)
	test_x = torch.from_numpy(X_all)
	test_y = torch.from_numpy(y_all)
	# get test accuracy
	m_finetune.getTestAccuracy(test_x, test_y)
	# get train_x, train_y
	Y, segments, maxlen_seg, n_files, seq_length = tools.getAllData(TRAIN_PATH)
	X_all, y_all, _ = tools.transferData(Y, segments, n_files, seq_length)
	train_x = torch.from_numpy(X_all)
	train_y = torch.from_numpy(y_all)
	train_data = torch.utils.data.TensorDataset(train_x, train_y)
	# trainAllLayers
	m_finetune.trainPartialLayers(train_data)
	# get test accuracy
	m_finetune.getTestAccuracy(test_x, test_y)

