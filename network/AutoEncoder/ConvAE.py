import os
import torch
from torch import nn
from torch.autograd import Variable

class ConvAE(nn.Module):

	def __init__(self, time_steps = 800, n_features = 3, use_cuda = False, params_pkl = "./params_ConvAE.pkl"):
		super(ConvAE, self).__init__()

		self.time_steps = time_steps
		self.n_features = n_features
		self.use_cuda = use_cuda
		self.params_pkl = params_pkl

		self.n_filters1 = self.n_features * 2
		self.kernel_size1 = 15
		self.scale_factor1 = 2
		self.n_filters2 = self.n_filters1 * 2
		self.kernel_size2 = 15
		self.scale_factor2 = 2
		self.n_filters3 = self.n_filters2 * 2
		self.kernel_size3 = 15
		self.scale_factor3 = 2

		self._build_net()

	def _build_net(self):
		self.encoder = nn.Sequential(
			# (batch, 3, 800)
			nn.Conv1d(in_channels = self.n_features, out_channels = self.n_filters1, kernel_size = self.kernel_size1, padding = self.kernel_size1 // 2),
			# (batch, 6, 800)
			nn.ReLU(True),
			nn.MaxPool1d(kernel_size = self.scale_factor1, stride = self.scale_factor1),
			# (batch, 6, 400)
			nn.Conv1d(in_channels = self.n_filters1, out_channels = self.n_filters2, kernel_size = self.kernel_size2, padding = self.kernel_size2 // 2),
			# (batch, 12, 400)
			nn.ReLU(True),
			nn.MaxPool1d(kernel_size = self.scale_factor2, stride = self.scale_factor2),
			# (batch, 12, 200)
			nn.Conv1d(in_channels = self.n_filters2, out_channels = self.n_filters3, kernel_size = self.kernel_size3, padding = self.kernel_size3 // 2),
			# (batch, 24, 200)
			nn.ReLU(True),
			nn.MaxPool1d(kernel_size = self.scale_factor3, stride = self.scale_factor3),
			# (batch, 24, 100)
		)

		self.decoder = nn.Sequential(
			# (batch, 24, 100)
			nn.Upsample(scale_factor = self.scale_factor3, mode = 'nearest'),
			# (batch, 24, 200)
			nn.Conv1d(in_channels = self.n_filters3, out_channels = self.n_filters2, kernel_size = self.kernel_size3, padding = self.kernel_size3 // 2),
			nn.ReLU(True),
			# (batch, 12, 200)
			nn.Upsample(scale_factor = self.scale_factor2, mode = 'nearest'),
			# (batch, 12, 400)
			nn.Conv1d(in_channels = self.n_filters2, out_channels = self.n_filters1, kernel_size = self.kernel_size2, padding = self.kernel_size2 // 2),
			nn.ReLU(True),
			# (batch, 6, 400)
			nn.Upsample(scale_factor = self.scale_factor1, mode = 'nearest'),
			# (batch, 6, 800)
			nn.Conv1d(in_channels = self.n_filters1, out_channels = self.n_features, kernel_size = self.kernel_size1, padding = self.kernel_size1 // 2),
			nn.ReLU(True),
			# (batch, 3, 800)
		)

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

	def forward(self, input):
		encode_output = self.encoder(input)
		# print(encode_output.shape)
		decode_output, encode_output = self.decoder(encode_output), None
		# print(decode_output.shape)
		output = decode_output
		return output

	def trainAllLayers(self, src_x, target_x, learning_rate = 0.001, n_epoches = 10, batch_size = 20, shuffle = True):
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
		conv_ae = ConvAE(use_cuda = use_cuda).cuda()
	else:
		conv_ae = ConvAE(use_cuda = use_cuda)
	train_x = torch.from_numpy(train_x)
	# train
	conv_ae.get_model(train_x, train_x)
