import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
# local network
import sys
sys.path.append("..")
from lstm import LSTMHardSigmoid

class SeqAE(nn.Module):

	def __init__(self, time_steps = 800, n_features = 3, use_cuda = False, params_pkl = "./params_SeqAE.pkl"):
		super(SeqAE, self).__init__()

		self.time_steps = time_steps
		self.n_features = n_features
		self.use_cuda = use_cuda
		self.params_pkl = params_pkl

		self.bidirectional = False
		self.n_layers = 1
		self.n_hidden = self.time_steps // 4

		self._build_net()

	def _build_net(self):
		self.encoder = LSTMHardSigmoid(input_size = self.n_features, hidden_size = self.n_hidden, num_layers = self.n_layers, dropout = 0.2, batch_first = True, bidirectional = self.bidirectional, bias = True)
		self.decoder = LSTMHardSigmoid(input_size = self.n_hidden, hidden_size = self.n_features, num_layers = self.n_layers, dropout = 0.2, batch_first = True, bidirectional = self.bidirectional, bias = True)

	def reset_parameters(self):
		# get weight set
		encoder_weights = ((name, params.data) for name, params in self.named_parameters() if (("encoder" in name) and ("weight" in name)))
		encoder_biases = ((name, params.data) for name, params in self.named_parameters() if (("encoder" in name) and ("bias" in name)))
		decoder_weights = ((name, params.data) for name, params in self.named_parameters() if (("decoder" in name) and ("weight" in name)))
		decoder_biases = ((name, params.data) for name, params in self.named_parameters() if (("decoder" in name) and ("bias" in name)))
		for name, params_data in encoder_weights:
			# print(name)
			nn.init.xavier_uniform_(params_data)
		for name, params_data in decoder_weights:
			# print(name)
			nn.init.xavier_uniform_(params_data)
		for name, params_data in encoder_biases:
			# print(name)
			nn.init.constant_(params_data, 0)
		for name, params_data in decoder_biases:
			# print(name)
			nn.init.constant_(params_data, 0)

	def init_hidden(self, batch_size):
		"""
		init blstm's hidden states
		"""
		if self.bidirectional:
			n_layers = self.n_layers * 2
		else:
			n_layers = self.n_layers
		# encoder_hidden
		if self.use_cuda:
			hidden_state = torch.zeros(n_layers, batch_size, self.n_hidden).cuda()
			cell_state = torch.zeros(n_layers, batch_size, self.n_hidden).cuda()
		else:
			hidden_state = torch.zeros(n_layers, batch_size, self.n_hidden)
			cell_state = torch.zeros(n_layers, batch_size, self.n_hidden)
		self.encoder_hidden = (hidden_state, cell_state)
		# decoder_hidden
		if self.use_cuda:
			hidden_state = torch.zeros(n_layers, batch_size, self.n_features).cuda()
			cell_state = torch.zeros(n_layers, batch_size, self.n_features).cuda()
		else:
			hidden_state = torch.zeros(n_layers, batch_size, self.n_features)
			cell_state = torch.zeros(n_layers, batch_size, self.n_features)
		self.decoder_hidden = (hidden_state, cell_state)

	def forward(self, input):
		# (batch, 3, 800)
		input = input.permute(0, 2, 1)
		# (batch, 800, 3)
		(encode_output, self.encoder_hidden) = self.encoder(input, self.encoder_hidden)
		print(encode_output.shape)
		# (batch, 800, 200)
		# MaxPooling1D time_steps
		encode_output = encode_output.permute(0, 2, 1)
		# (batch, 200, 800)
		maxPooling_output, encode_output = F.max_pool1d(encode_output, kernel_size = encode_output.size(2)).squeeze(2), None
		# (batch, 200)
		maxPooling_output = maxPooling_output.view(-1, 1, self.n_hidden)
		# (batch, 1, 200)
		maxPooling_output = maxPooling_output.repeat(1, self.time_steps, 1)
		# (batch, 800, 200)
		(decode_output, self.decoder_hidden), maxPooling_output = self.decoder(maxPooling_output, self.decoder_hidden), None
		print(decode_output.shape)
		# (batch, 800, 3)
		output, decode_output = decode_output.permute(0, 2, 1), None
		# (batch, 3, 800)
		return output

	def trainAllLayers(self, src_x, target_x, learning_rate = 0.01, n_epoches = 10, batch_size = 20, shuffle = True):
		# optimize all cnn parameters
		optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
		# the target label is not one-hotted
		loss_func = nn.MSELoss()

		# init params
		self.reset_parameters()

		# load params
		self.load_params()

		# set train mode True
		self.train()

		# if use_cuda
		if self.use_cuda:
			src_x = src_x.cuda()
			target_y = target_y.cuda()

		# get train_data
		train_data = torch.utils.data.TensorDataset(src_x, target_x)
		# Data Loader for easy mini-batch return in training
		train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = shuffle)

		# training and testing
		for epoch in range(n_epoches):
			# init loss & acc
			train_loss = 0
			for step, (b_x, b_y) in enumerate(train_loader):		# gives batch data
				b_x = b_x.view(-1, self.n_features, self.time_steps)	# reshape x to (batch, n_features, time_step)
				if self.use_cuda:
					b_x, b_y = Variable(b_x).cuda(), Variable(b_y).cuda()
				else:
					b_x, b_y = Variable(b_x), Variable(b_y)
				# init hidden
				self.init_hidden(b_x.size(0))
				# get output
				output = self(b_x)									# AE output
				# get loss
				loss = loss_func(output, b_y)						# MSE loss
				train_loss += loss.item() * len(b_y)
				# backward
				optimizer.zero_grad()								# clear gradients for this training step
				loss.backward()										# backpropagation, compute gradients
				optimizer.step()									# apply gradients

				# print loss
				if (step + 1) % 1 == 0:
					print("[{}/{}], train loss is: {:.6f}".format(step, len(train_loader), train_loss / ((step + 1) * batch_size)))

			# save params
			self.save_params()

		print("train finish!")

	def save_params(self):
		"""
		save params & adabn's inner stats
		"""
		torch.save(self.state_dict(), self.params_pkl)
		print("save_params success!")

	def load_params(self):
		"""
		load params & adabn's inner stats
		"""
		if os.path.exists(self.params_pkl):
			if self.use_cuda:
				self.load_state_dict(torch.load(self.params_pkl, map_location = torch.device('cuda')))
			else:
				self.load_state_dict(torch.load(self.params_pkl, map_location = torch.device('cpu')))
			print("load_params success!")

	def get_model(self, src_x = None, target_x = None, n_epoches = 20, pre_trained = False):
		"""
		get pretrained model
		"""
		if pre_trained:
			self.load_params()
		else:
			self.trainAllLayers(src_x, target_x, n_epoches = n_epoches)
		return self

if __name__ == "__main__":
	torch.manual_seed(1)

	import sys
	sys.path.append("../..")
	import tools
	PATH_SYS = sys.argv[1]
	# get train_x, train_y
	Y, segments, maxlen_seg, n_files, seq_length = tools.getAllData(PATH_SYS)
	train_x, train_y, _ = tools.transferData(Y, segments, n_files, seq_length)
	# whether use cuda
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		seq_ae = SeqAE(use_cuda = use_cuda).cuda()
	else:
		seq_ae = SeqAE(use_cuda = use_cuda)
	train_x = torch.from_numpy(train_x)
	# train
	seq_ae.get_model(train_x, train_x)
