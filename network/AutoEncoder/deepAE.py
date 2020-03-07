import torch
from torch import nn

class deepAE(nn.Module):

	def __init__(self, time_steps = 800, n_features = 3, use_cuda = False, params_pkl = "./params_deepAE.pkl"):
		super(deepAE, self).__init__()

		self.time_steps = time_steps
		self.n_features = n_features
		self.use_cuda = use_cuda
		self.params_pkl = params_pkl

		self.N_FC0 = self.time_steps * self.n_features
		self.N_FC1 = self.N_FC0 // 4
		self.N_FC2 = self.N_FC1 // 4
		self.N_FC3 = self.N_FC2 // 4

		self._build_net()

	def _build_net(self):
		self.encoder = nn.Sequential(
			nn.Linear(self.N_FC0, self.N_FC1, bias = True),
			nn.ReLU(True),
			nn.Linear(self.N_FC1, self.N_FC2, bias = True),
			nn.ReLU(True),
			nn.Linear(self.N_FC2, self.N_FC3, bias = True),
			nn.ReLU(True)
		)

		self.decoder = nn.Sequential(
			nn.Linear(self.N_FC3, self.N_FC2, bias = True),
			nn.ReLU(True),
			nn.Linear(self.N_FC2, self.N_FC1, bias = True),
			nn.ReLU(True),
			nn.Linear(self.N_FC1, self.N_FC0, bias = True),
			nn.ReLU(True)
			# nn.Sigmoid(True)
		)

	def init_weights(self):
		for name, params in net.named_parameters():
			print(name, params)

	def forward(self.input):
		input = input.permute(0, 2, 1).contiguous()		# (batch, 800, 3)
		input = input.view(input.shape[0], -1)
		encode_output = self.encoder(input)
		decode_output, encode_output = self.decoder(encode_output), None
		output, decode_output = decode_output.view(decode_output.shape[0], self.time_steps, self.n_features), None
		output = output.permute(0, 2, 1).contiguous()	# (batch, 3, 800)
		return output

	def train(self, src_x, target_x, learning_rate = 0.01, n_epoches = 10, batch_size = 20, shuffle = True):
		# optimize all cnn parameters
		optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
		# the target label is not one-hotted
		loss_func = nn.MSELoss()

		# init params
		self.init_weights()

		# load params
		self.load_params()

		# set train mode True
		self.train()

		# if use_cuda
		if self.use_cuda:
			src_x = src_x.cuda()
			target_y = target_y.cuda()

		# get train_data
		train_data = torch.utils.data.TensorDataset(src_x, target_y)
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
				output = self(b_x)									# CNN_BLSTM output
				# get loss
				loss = loss_func(output, b_y)						# cross entropy loss
				train_loss += loss.item() * len(b_y)
				# backward
				optimizer.zero_grad()								# clear gradients for this training step
				loss.backward()										# backpropagation, compute gradients
				optimizer.step()									# apply gradients

				# print loss
				if (step + 1) % 10 == 0:
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
		if os.path.exists(os.path.join(self.params_dir, cnnblstm_with_adabn.PARAMS_FILE)):
			if self.use_cuda:
				self.load_state_dict(torch.load(self.params_pkl, map_location = torch.device('cuda')))
			else:
				self.load_state_dict(torch.load(self.params_pkl, map_location = torch.device('cpu')))
			print("load_params success!")

	def get_model(self, src_x, target_x, pre_trained = False):
		"""
		get pretrained model
		"""
		if pre_trained:
			self.load_params()
		else:
			self.train(src_x, target_x)
		return self

if __name__ == "__main__":
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
		deep_ae = deepAE(use_cuda = use_cuda).cuda()
	else:
		deep_ae = deepAE(use_cuda = use_cuda)
	train_x = torch.from_numpy(train_x)
	# train
	deep_ae.train(train_x, train_x)
