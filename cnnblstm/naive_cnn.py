import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
# local model
import sys
sys.path.append("../network")
from lstm import LSTMHardSigmoid

class naive_cnn(nn.Module):

	def __init__(self, time_steps = 800, n_features = 3, n_outputs = 10, params_file = "./params.pkl", use_cuda = False):
		super(naive_cnn, self).__init__()

		self.time_steps = time_steps
		self.n_features = n_features
		self.n_outputs = n_outputs
		self.params_file = params_file

		self.n_filters = 128
		self.kernel_size = 15

		self.use_cuda = use_cuda

		# build net1 features.cnn
		self.net1 = nn.Sequential(
			# (batch, 3, 800)
			nn.Conv1d(in_channels = self.n_features, out_channels = self.n_filters, kernel_size = self.kernel_size),
			nn.ReLU(),
			# nn.Sigmoid(),
			nn.Dropout(p = 0.5),
			# (batch, 128, 786)
			nn.MaxPool1d(kernel_size = 2)
			# (batch, 128, 393)
		)

		# build net2 classifier(fc->fc)
		self.net3 = nn.Sequential(
			# nn.Tanh(),
			nn.Linear(((self.time_steps - self.kernel_size + 1) // 2), 50, bias = True),
			nn.ReLU(),
			# nn.Sigmoid(),
			nn.Dropout(p = 0.2),
			nn.Linear(50, self.n_outputs, bias = True),
			nn.Softmax(dim = 1)
		)

	def reset_parameters(self):
		"""
		temp useless
		Here we reproduce Keras default initialization weights for consistency with Keras version
		"""
		# get weights & bias set
		net1_weights = ((name, param.data) for name, param in self.named_parameters() if (("weight" in name) and (("net1" in name) and ("net1_adabn" not in name))))
		net1_biases = ((name, param.data) for name, param in self.named_parameters() if (("bias" in name) and (("net1" in name) and ("net1_adabn" not in name))))
		net3_weights = ((name, param.data) for name, param in self.named_parameters() if (("weight" in name) and (("net3" in name) and ("net3_adabn" not in name))))
		net3_biases = ((name, param.data) for name, param in self.named_parameters() if (("bias" in name) and (("net3" in name) and ("net3_adabn" not in name))))
		# init weights & bias
		for name, params_data in net1_weights:
			nn.init.xavier_uniform_(params_data)
		for name, params_data in net1_biases:
			nn.init.constant_(params_data, 0)
		for name, params_data in net3_weights:
			nn.init.xavier_uniform_(params_data)
		for name, params_data in net3_biases:
			nn.init.constant_(params_data, 0)

	def forward(self, input):
		"""
		compute the output of input according to the entire network model
		"""
		# MaxPool1d
		maxPool1d_output = self.net1(input)
		# (batch, 128, 393)
		maxPool1d_t_output = maxPool1d_output.permute(0, 2, 1).contiguous()
		# (batch, 393, 128)
		# BiLSTM
		maxPooling_output = F.max_pool1d(maxPool1d_t_output, kernel_size = maxPool1d_t_output.size(2)).squeeze(2)
		# get classifier
		linear2_softmax_output = self.net3(maxPooling_output)

		return linear2_softmax_output

	def trainAllLayers(self, train_x, train_y, learning_rate = 0.01, n_epoches = 20, batch_size = 20, shuffle = True):
		"""
		train all layers of network model
		"""
		# get train_data
		train_data = torch.utils.data.TensorDataset(train_x, train_y)
		# Data Loader for easy mini-batch return in training
		train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = shuffle)
		# optimize all cnn parameters
		optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
		# the target label is not one-hotted
		loss_func = nn.CrossEntropyLoss()

		# init params
		self.reset_parameters()

		# load params
		self.load_params()

		# set train mode True
		self.train()

		# get parallel model
		parallel_cnn = self
		if self.use_cuda:
			# print("we use cuda!")
			parallel_cnn = torch.nn.DataParallel(self, device_ids = range(torch.cuda.device_count()))
			# parallel_cnn = parallel_cnn.cuda()

		# training and testing
		for epoch in range(n_epoches):
			# init loss & acc
			train_loss = 0
			train_acc = 0
			for step, (b_x, b_y) in enumerate(train_loader):		# gives batch data
				b_x = b_x.view(-1, self.n_features, self.time_steps)	# reshape x to (batch, n_features, time_step)
				if self.use_cuda:
					b_x, b_y = Variable(b_x).cuda(), Variable(b_y).cuda()
				else:
					b_x, b_y = Variable(b_x), Variable(b_y)
				# get output
				output = parallel_cnn(b_x)							# CNN_BLSTM output
				# get loss
				loss = loss_func(output, b_y)						# cross entropy loss
				train_loss += loss.item() * len(b_y)
				_, pre = torch.max(output, 1)
				num_acc = (pre == b_y).sum()
				train_acc += num_acc.item()
				# backward
				optimizer.zero_grad()								# clear gradients for this training step
				loss.backward()										# backpropagation, compute gradients
				optimizer.step()									# apply gradients

				# print loss
				# if (step + 1) % 10 == 0:
				# 	print("[{}/{}], train loss is: {:.6f}, train acc is: {:.6f}".format(step, len(train_loader), train_loss / ((step + 1) * batch_size), train_acc / ((step + 1) * batch_size)))
			print("[{}/{}], train loss is: {:.6f}, train acc is: {:.6f}".format(len(train_loader), len(train_loader), train_loss / (len(train_loader) * batch_size), train_acc / (len(train_loader) * batch_size)))

			# save params
			self.save_params()

		print("train finish!")

	def getTestAccuracy(self, test_x, test_y):
		"""
		test network model with test set
		"""
		# init params
		self.reset_parameters()

		# load params
		self.load_params()

		# set eval
		self.eval()

		# get parallel model
		parallel_cnn = self
		if self.use_cuda:
			# print("we use cuda!")
			parallel_cnn = torch.nn.DataParallel(self, device_ids = range(torch.cuda.device_count()))
			# parallel_cnn = parallel_cnn.cuda()

		with torch.no_grad():
			if self.use_cuda:
				test_x, test_y = Variable(test_x).cuda(), Variable(test_y).cuda()
			else:
				test_x, test_y = Variable(test_x), Variable(test_y)
		# get output
		output = parallel_cnn(test_x)
		# print(output)
		prediction = torch.max(output, 1)[1]
		pred_y = prediction.cpu().data.numpy()
		# print(pred_y)
		target_y = test_y.cpu().data.numpy()
		# print(test_y)

		accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
		print("Accuracy: ", str(accuracy))

	def save_params(self):
		"""
		save params
		"""
		torch.save(self.state_dict(), self.params_file)
		print("save_params success!")

	def load_params(self):
		"""
		load params
		"""
		if os.path.exists(self.params_file):
			if self.use_cuda:
				self.load_state_dict(torch.load(self.params_file, map_location = torch.device('cuda')))
			else:
				self.load_state_dict(torch.load(self.params_file, map_location = torch.device('cpu')))
			print("load_params success!")

	def get_model(self, pre_trained = False):
		"""
		get pretrained model
		"""
		if pre_trained:
			self.load_params()
		return self

if __name__ == '__main__':
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		naive_cnn = naive_cnn(use_cuda = use_cuda).cuda()
	else:
		naive_cnn = naive_cnn(use_cuda = use_cuda)
	print(naive_cnn)
	# get train_x, train_y
	train_x = torch.rand(20, 3, 800, dtype = torch.float32)
	train_y = torch.randint(10, (20, ), dtype = torch.int64)
	# train_y = torch.LongTensor(20, 1).random_() % 10
	print(train_x.type())
	# train_y = torch.zeros(20, 10).scatter_(1, train_y, 1)
	print(train_y)
	train_data = torch.utils.data.TensorDataset(train_x, train_y)
	naive_cnn.trainAllLayers(train_data)
