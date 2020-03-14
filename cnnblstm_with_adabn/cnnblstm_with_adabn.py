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
import Coral
from lstm import LSTMHardSigmoid
from AdaBN import AdaBN
sys.path.append("../network/AutoEncoder")
import AutoEncoder

class cnnblstm_with_adabn(nn.Module):
	PARAMS_FILE = "params.pkl"
	PARAMS_AE = "params_ae.pkl"
	NET1_ADABN = "net1_adabn"
	NET2_ADABN = "net2_adabn"
	NET3_ADABN = "net3_adabn"

	def __init__(self, time_steps = 800, n_features = 3, n_outputs = 10, use_cuda = False, params_dir = "./params", enable_CORAL = False):
		super(cnnblstm_with_adabn, self).__init__()

		self.time_steps = time_steps
		self.n_features = n_features
		self.n_outputs = n_outputs
		self.use_cuda = use_cuda
		self.params_dir = params_dir
		if not os.path.exists(self.params_dir):
			os.mkdir(self.params_dir)
		self.enable_CORAL = enable_CORAL

		self.n_filters = 128
		self.kernel_size = 15
		self.n_hidden = 150	# 150
		self.n_layers = 1
		self.bidirectional = True

		self.ae = AutoEncoder.load_AE(type = "ConvAE", time_steps = self.time_steps, n_features = self.n_features, use_cuda = self.use_cuda, params_pkl = os.path.join(self.params_dir, cnnblstm_with_adabn.PARAMS_AE))

		# build net1 cnn
		self.net1 = nn.Sequential(
			# nn.Conv1d(in_channels = self.n_features, out_channels = self.n_filters, kernel_size = self.kernel_size),
			nn.Conv1d(in_channels = self.ae.n_filters3, out_channels = self.n_filters, kernel_size = self.kernel_size),
			nn.ReLU(),
			# nn.Sigmoid(),
			nn.Dropout(p = 0.5),
			nn.MaxPool1d(kernel_size = 2)
		)

		# build net1_adabn
		self.net1_adabn = AdaBN(self.n_filters, variables_dir = os.path.join(self.params_dir, cnnblstm_with_adabn.NET1_ADABN), use_cuda = self.use_cuda)

		# build net2 blstm
		# self.net2 = nn.LSTM(input_size = self.n_filters, hidden_size = self.n_hidden, num_layers = self.n_layers, dropout = 0.2, batch_first = True, bidirectional = self.bidirectional, bias = True)
		self.net2 = LSTMHardSigmoid(input_size = self.n_filters, hidden_size = self.n_hidden, num_layers = self.n_layers, dropout = 0.2, batch_first = True, bidirectional = self.bidirectional, bias = True)

		# build net2_adabn
		if self.bidirectional:
			n_blstm_output = self.n_hidden * 2
		else:
			n_blstm_output = self.n_hidden
		self.net2_adabn = AdaBN(n_blstm_output, variables_dir = os.path.join(self.params_dir, cnnblstm_with_adabn.NET2_ADABN), use_cuda = self.use_cuda)

		# build net3 fc
		self.net3 = nn.Sequential(
			nn.Linear(n_blstm_output, 50, bias = True),
			nn.ReLU(),
			# nn.Sigmoid(),
		)

		# build net3_adabn
		self.net3_adabn = AdaBN(50, variables_dir = os.path.join(self.params_dir, cnnblstm_with_adabn.NET3_ADABN), use_cuda = self.use_cuda)

		# build net4 fc
		self.net4 = nn.Sequential(
			nn.Dropout(p = 0.2),
			nn.Linear(50, self.n_outputs, bias = True),
			nn.Softmax(dim = 1)
		)

	def init_hidden(self, batch_size):
		"""
		init blstm's hidden states
		"""
		if self.bidirectional:
			n_layers = self.n_layers * 2
		else:
			n_layers = self.n_layers
		if self.use_cuda:
			hidden_state = torch.zeros(n_layers, batch_size, self.n_hidden).cuda()
			cell_state = torch.zeros(n_layers, batch_size, self.n_hidden).cuda()
		else:
			hidden_state = torch.zeros(n_layers, batch_size, self.n_hidden)
			cell_state = torch.zeros(n_layers, batch_size, self.n_hidden)
		self.hidden = (hidden_state, cell_state)

	def reset_parameters(self):
		"""
		temp useless
		Here we reproduce Keras default initialization weights for consistency with Keras version
		"""
		# get weights & bias set
		net1_weights = ((name, param.data) for name, param in self.named_parameters() if (("weight" in name) and (("net1" in name) and ("net1_adabn" not in name))))
		net1_biases = ((name, param.data) for name, param in self.named_parameters() if (("bias" in name) and (("net1" in name) and ("net1_adabn" not in name))))
		# net2_weights = ((name, param.data) for name, param in self.named_parameters() if (("weight" in name) and (("net2" in name) and ("net2_adabn" not in name))))
		# net2_biases = ((name, param.data) for name, param in self.named_parameters() if (("bias" in name) and (("net2" in name) and ("net2_adabn" not in name))))
		net3_weights = ((name, param.data) for name, param in self.named_parameters() if (("weight" in name) and (("net3" in name) and ("net3_adabn" not in name))))
		net3_biases = ((name, param.data) for name, param in self.named_parameters() if (("bias" in name) and (("net3" in name) and ("net3_adabn" not in name))))
		net4_weights = ((name, param.data) for name, param in self.named_parameters() if (("weight" in name) and (("net4" in name) and ("net4_adabn" not in name))))
		net4_biases = ((name, param.data) for name, param in self.named_parameters() if (("bias" in name) and (("net4" in name) and ("net4_adabn" not in name))))
		# init weights & bias
		self.ae.reset_parameters()
		for name, params_data in net1_weights:
			# print(name)
			nn.init.xavier_uniform_(params_data)
		for name, params_data in net1_biases:
			nn.init.constant_(params_data, 0)
		self.net1_adabn.reset_parameters()
		self.net2.reset_parameters()		# lstm reset parameters
		self.net2_adabn.reset_parameters()
		for name, params_data in net3_weights:
			nn.init.xavier_uniform_(params_data)
		for name, params_data in net3_biases:
			nn.init.constant_(params_data, 0)
		self.net3_adabn.reset_parameters()
		for name, params_data in net4_weights:
			nn.init.xavier_uniform_(params_data)
		for name, params_data in net4_biases:
			nn.init.constant_(params_data, 0)

	def forward(self, input):
		"""
		compute the output of input according to the entire network model
		"""
		# print(input.shape)
		# AutoEncoder
		input = self.ae.encoder(input)
		# input = self.ae(input)
		# MaxPool1d
		maxPool1d_output = self.net1(input)
		# maxPool1d_adabn_output = maxPool1d_output
		maxPool1d_adabn_output, maxPool1d_output = self.net1_adabn(maxPool1d_output), None
		maxPool1d_adabn_t_output = maxPool1d_adabn_output.permute(0, 2, 1).contiguous()
		# BiLSTM
		(bilstm_output, _), maxPool1d_adabn_t_output = self.net2(maxPool1d_adabn_t_output, None), None
		# MaxPooling1D time_steps
		bilstm_output = bilstm_output.permute(0, 2, 1)
		maxPooling_output, bilstm_output = F.max_pool1d(bilstm_output, kernel_size = bilstm_output.size(2)).squeeze(2), None
		# maxPooling_adabn_output = maxPooling_output
		maxPooling_adabn_output, maxPooling_output = self.net2_adabn(maxPooling_output), None
		# get classifier
		net3_output, maxPooling_adabn_output = self.net3(maxPooling_adabn_output), None
		net3_adabn_output, net3_output = self.net3_adabn(net3_output), None
		linear2_softmax_output, net3_adabn_output = self.net4(net3_adabn_output), None

		return linear2_softmax_output

	def update_adabn_running_stats(self):
		"""
		update adabn running states, update mu_j with mu_j_next to start next round
		"""
		self.net1_adabn.update_running_stats()
		self.net2_adabn.update_running_stats()
		self.net3_adabn.update_running_stats()

	def trainAllLayers(self, train_x, train_y, test_x = None, learning_rate = 0.01, n_epoches = 20, batch_size = 20, shuffle = True):
		"""
		train all layers of network model
		"""
		# print(os.environ["CUDA_VISIBLE_DEVICES"])
		# CORAL
		if self.enable_CORAL:
			if test_x == None:
				print("ERROR: (in cnnblstm_with_adabn.trainAllLayers) test_x == None!")
				return
			# review train_x & test_x
			train_x = train_x.view(-1, self.time_steps * self.n_features)
			test_x = test_x.view(-1, self.time_steps * self.n_features)
			# get CORAL(train_x, test_x)
			train_x = Coral.CORAL_torch(train_x, test_x)
		# review train_x
		train_x = train_x.view(-1, self.n_features, self.time_steps)
		# optimize all cnn parameters
		params = [{"params": model.parameters()} for model in self.children() if model  not in [self.ae]]
		optimizer = torch.optim.Adam(params, lr = learning_rate)
		# the target label is not one-hotted
		loss_func = nn.CrossEntropyLoss()

		# init params
		self.reset_parameters()

		# load params
		self.load_params()

		# set train mode True
		self.train()

		# get parallel model
		parallel_cba = self
		if self.use_cuda:
			# print("we use cuda!")
			parallel_cba = torch.nn.DataParallel(self, device_ids = range(torch.cuda.device_count()))
			# parallel_cba = parallel_cba.cuda()

		# if use_cuda
		if self.use_cuda:
			train_x = train_x.cuda()
			train_y = train_y.cuda()

		# get autoencoder
		self.ae = AutoEncoder.train_AE(self.ae, train_x, train_x, n_epoches = 20)
		self.ae.save_params()
 
		# get train_data
		train_data = torch.utils.data.TensorDataset(train_x, train_y)
		# Data Loader for easy mini-batch return in training
		train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = shuffle)

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
				"""
				# get hidden
				if self.use_cuda:
					self.init_hidden(b_x.size(0) // torch.cuda.device_count())
				else:
					self.init_hidden(b_x.size(0))
				"""
				# update adabn running stats
				self.update_adabn_running_stats()
				# get output
				output = parallel_cba(b_x)									# CNN_BLSTM output
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
				# if (step + 1) % 5 == 0:
					# print("[{}/{}], train loss is: {:.6f}, train acc is: {:.6f}".format(step, len(train_loader), train_loss / ((step + 1) * batch_size), train_acc / ((step + 1) * batch_size)))
			print("[{}/{}], train loss is: {:.6f}, train acc is: {:.6f}".format(len(train_loader), len(train_loader), train_loss / (len(train_loader) * batch_size), train_acc / (len(train_loader) * batch_size)))

			# save params
			self.save_params()

		# print("train finish!")

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
		parallel_cba = self
		if self.use_cuda:
			# print("we use cuda!")
			parallel_cba = torch.nn.DataParallel(self, device_ids = range(torch.cuda.device_count()))
			# parallel_cba = parallel_cba.cuda()
		# cuda test_data
		with torch.no_grad():
			if self.use_cuda:
				test_x, test_y = Variable(test_x).cuda(), Variable(test_y).cuda()
			else:
				test_x, test_y = Variable(test_x), Variable(test_y)
		"""
		# get hidden
		if self.use_cuda:
			self.init_hidden(test_x.size(0) // torch.cuda.device_count())
		else:
			self.init_hidden(test_x.size(0))
		"""
		# update adabn running stats
		self.update_adabn_running_stats()
		# get output
		with torch.no_grad():
			output = parallel_cba(test_x)
		# print(output)
		prediction = torch.max(output, 1)[1]
		pred_y = prediction.cpu().data.numpy()
		# print(pred_y)
		target_y = test_y.cpu().data.numpy()
		# print(test_y)

		accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
		# print("Accuracy: ", str(accuracy))
		return accuracy

	def save_params(self):
		"""
		save params & adabn's inner stats
		"""
		self.save_adabn_variables()
		torch.save(self.state_dict(), os.path.join(self.params_dir, cnnblstm_with_adabn.PARAMS_FILE))
		self.ae.save_params()
		# print("save_params success!")

	def save_adabn_variables(self):
		"""
		save adabn's inner stats
		"""
		self.net1_adabn.save_attrs()
		self.net2_adabn.save_attrs()
		self.net3_adabn.save_attrs()

	def load_params(self):
		"""
		load params & adabn's inner stats
		"""
		self.load_adabn_variables()
		if os.path.exists(os.path.join(self.params_dir, cnnblstm_with_adabn.PARAMS_FILE)):
			if self.use_cuda:
				self.load_state_dict(torch.load(os.path.join(self.params_dir, cnnblstm_with_adabn.PARAMS_FILE), map_location = torch.device('cuda')))
			else:
				self.load_state_dict(torch.load(os.path.join(self.params_dir, cnnblstm_with_adabn.PARAMS_FILE), map_location = torch.device('cpu')))
			# print("load_params success!")
		self.ae.load_params()

	def load_adabn_variables(self):
		"""
		load adabn's inner stats
		"""
		self.net1_adabn.load_attrs()
		self.net2_adabn.load_attrs()
		self.net3_adabn.load_attrs()

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
		cnnblstm = cnnblstm_with_adabn(use_cuda = use_cuda).cuda()
	else:
		cnnblstm = cnnblstm_with_adabn(use_cuda = use_cuda)
	print(cnnblstm)
	# get train_x, train_y
	train_x = torch.rand(20, 3, 800, dtype = torch.float32)
	train_y = torch.randint(10, (20, ), dtype = torch.int64)
	# train_y = torch.LongTensor(20, 1).random_() % 10
	print(train_x.type())
	# train_y = torch.zeros(20, 10).scatter_(1, train_y, 1)
	print(train_y)
	train_data = torch.utils.data.TensorDataset(train_x, train_y)
	cnnblstm.trainAllLayers(train_data)
