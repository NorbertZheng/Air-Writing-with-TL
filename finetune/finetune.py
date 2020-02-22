import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# local class
import sys
sys.path.append("../cnnblstm")
from cnnblstm import cnnblstm

class finetune(nn.Module):

	def __init__(self, time_steps = 800, n_features = 3, n_outputs = 10, params_file = "./params.pkl", use_cuda = 0, finetune_params_file = "./finetune_params.pkl"):
		super(finetune, self).__init__()

		self.finetune_params_file = finetune_params_file

		self.time_steps = time_steps
		self.n_features = n_features
		self.n_outputs = n_outputs
		self.use_cuda = use_cuda

		# get pretrained cnnblstm model
		cnn_blstm = cnnblstm(time_steps = time_steps, n_features = n_features, n_outputs = n_outputs, params_file = params_file, use_cuda = use_cuda).get_model(pre_trained = True).get_model(pretrained = True)
		# set cnnblstm's net3(classifier) with None
		cnn_blstm.net3 = nn.Sequential()
		self.features = cnn_blstm
		# rebuild finetune's classifier
		if self.features.bidirectional:
			n_feature_inputs = self.features.n_hidden * 2
		else:
			n_feature_inputs = self.features.n_hidden
		self.classifier = nn.Sequential(
			# nn.Tanh(),
			nn.Linear(n_feature_inputs, 50, bias = True),
			nn.ReLU(),
			# nn.Sigmoid(),
			nn.Dropout(p = 0.2),
			nn.Linear(50, self.n_outputs, bias = True),
			nn.Softmax(dim = 1)
		)

	def forward(self, input):
		"""
		compute the output of input according to the entire network model
		"""
		# get features
		features_output = self.features(input)
		features_output = features_output.view(features_output.size(0), -1)
		# get classifier
		classifier_output = self.classifier(features_output)

		return classifier_output

	def init_partial_weights(self):
		"""
		Here we reproduce Keras default initialization weights for consistency with Keras version
		"""
		ih = (param.data for name, param in self.classifier.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.classifier.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.classifier.named_parameters() if 'bias' in name)
		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)

	def trainPartialLayers(self, train_data, learning_rate = 0.001, n_epoches = 10, batch_size = 20, shuffle = True):
		"""
		train classifier layers of network model
		"""
		# Data Loader for easy mini-batch return in training
		train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = shuffle)
		# optimize all cnn parameters
		params = [{"params": model.parameters()} for model in self.children() if model in [self.classifier]]
		optimizer = torch.optim.Adam(params, lr = learning_rate)
		# the target label is not one-hotted
		loss_func = nn.CrossEntropyLoss()

		# init params
		self.init_partial_weights()

		# load params
		self.load_params()

		# set train mode True
		self.train()

		# training and testing
		for epoch in range(n_epoches):
			# init loss & acc
			train_loss = 0
			train_acc = 0
			for step, (b_x, b_y) in enumerate(train_loader):		# gives batch data
				b_x = b_x.view(-1, self.n_features, self.time_steps)	# reshape x to (batch, n_features, time_step)
				if self.use_cuda == 1:
					b_x, b_y = Variable(b_x).cuda(), Variable(b_y).cuda()
				else:
					b_x, b_y = Variable(b_x), Variable(b_y)
				# get hidden
				self.features.init_hidden(b_x.size(0))
				# get output
				output = self(b_x)									# CNN_BLSTM output
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
				if (step + 1) % 10 == 0:
					print("[{}/{}], train loss is: {:.6f}, train acc is: {:.6f}".format(step, len(train_loader), train_loss / ((step + 1) * batch_size), train_acc / ((step + 1) * batch_size)))

			# save params
			self.save_params()

		print("train finish!")

	def getTestAccuracy(self, test_x, test_y):
		"""
		test network model with test set
		"""
		# init params
		self.init_partial_weights()

		# load params
		self.load_params()

		# set eval
		self.eval()

		with torch.no_grad():
			if self.use_cuda == 1:
				test_x, test_y = Variable(test_x).cuda(), Variable(test_y).cuda()
			else:
				test_x, test_y = Variable(test_x), Variable(test_y)
		# get hidden
		self.features.init_hidden(test_x.size(0))
		# get output
		output = self(test_x)
		print(output)
		prediction = torch.max(output, 1)[1]
		if self.use_cuda == 1:
			prediction = prediction.cpu()
			test_y = test_y.cpu()
		pred_y = prediction.cpu().data.numpy()
		print(pred_y)
		target_y = test_y.cpu().data.numpy()
		print(test_y)

		accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
		print("Accuracy: ", str(accuracy))

	def save_params(self):
		"""
		save params
		"""
		torch.save(self.state_dict(), self.finetune_params_file)
		print("save_params success!")

	def load_params(self):
		"""
		load params
		"""
		if os.path.exists(self.finetune_params_file):
			if self.use_cuda == 0:
				self.load_state_dict(torch.load(self.finetune_params_file, map_location = torch.device('cpu')))
			else:
				self.load_state_dict(torch.load(self.finetune_params_file, map_location = torch.device('cuda')))
			print("load_params success!")

if __name__ == '__main__':
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		m_finetune = finetune(use_cuda = 1).cuda()
	else:
		m_finetune = finetune(use_cuda = 0)
	print(m_finetune)
	# get train_x, train_y
	train_x = torch.rand(20, 3, 800, dtype = torch.float32)
	train_y = torch.randint(10, (20, ), dtype = torch.int64)
	# train_y = torch.LongTensor(20, 1).random_() % 10
	print(train_x.type())
	# train_y = torch.zeros(20, 10).scatter_(1, train_y, 1)
	print(train_y)
	train_data = torch.utils.data.TensorDataset(train_x, train_y)
	m_finetune.trainPartialLayers(train_data)

