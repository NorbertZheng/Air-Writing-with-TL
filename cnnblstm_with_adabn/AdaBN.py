from __future__ import division

import os
import pickle
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class AdaBN(nn.Module):
	MU_J_FILE = "mu_j.pkl"
	SIGMA_J_FILE = "sigma_j.pkl"
	N_J_FILE = "n_j.pkl"
	MU_J_NEXT_FILE = "mu_j_next.pkl"
	SIGMA_J_NEXT_FILE = "sigma_j_next.pkl"
	N_J_NEXT_FILE = "n_j_next.pkl"

	def __init__(self, n_features, eps = 1e-5, affine = True, variables_dir = "./variables", use_cuda = False):
		super(AdaBN, self).__init__()

		self.n_features = n_features
		self.eps = eps
		self.affine = affine
		self.variables_dir = variables_dir
		if not os.path.exists(self.variables_dir):
			os.mkdir(self.variables_dir)
		self.use_cuda = use_cuda

		if self.affine:
			self.weight = Parameter(torch.Tensor(self.n_features), requires_grad = True)
			self.bias = Parameter(torch.Tensor(self.n_features), requires_grad = True)
		else:
			self.weight = torch.ones(self.n_features, dtype = torch.float32)
			self.bais = torch.zeros(self.n_features, dtype = torch.float32)
		self.mu_j = Variable(torch.zeros(self.n_features, dtype = torch.float32))
		self.sigma_j = Variable(torch.ones(self.n_features, dtype = torch.float32))
		self.n_j = Variable(torch.zeros(1, dtype = torch.long))
		self.mu_j_next = Variable(torch.zeros(self.n_features, dtype = torch.float32))
		self.sigma_j_next = Variable(torch.ones(self.n_features, dtype = torch.float32))
		self.n_j_next = Variable(torch.zeros(1, dtype = torch.long))

		self.reset_parameters()

	def reset_parameters(self):
		self.reset_running_stats()
		if self.affine:
			nn.init.uniform_(self.weight)
			nn.init.zeros_(self.bias)
		else:
			nn.init.ones_(self.weight)
			nn.init.zeros_(self.bias)

	def reset_running_stats(self):
		self.mu_j = Variable(torch.zeros(self.n_features, dtype = torch.float32))
		self.sigma_j = Variable(torch.ones(self.n_features, dtype = torch.float32))
		self.n_j = Variable(torch.zeros(1, dtype = torch.long))
		self.mu_j_next = Variable(torch.zeros(self.n_features, dtype = torch.float32))
		self.sigma_j_next = Variable(torch.ones(self.n_features, dtype = torch.float32))
		self.n_j_next = Variable(torch.zeros(1, dtype = torch.long))

	def update_running_stats(self):
		'''
		self.mu_j.data.scatter_(dim = 0, index = torch.zeros(self.n_features).type(torch.long), src = self.mu_j_next.data)
		self.sigma_j.data.scatter_(dim = 0, index = torch.zeros(self.n_features).type(torch.long), src = self.sigma_j_next.data)
		self.n_j.data.scatter_(dim = 0, index = torch.zeros(1).type(torch.long), src = self.n_j_next.data)
		'''
		self.mu_j = self.mu_j_next
		self.sigma_j = self.sigma_j_next
		self.n_j = self.n_j_next		

	def set_running_stats(self, mu_j, sigma_j, n_j, mu_j_next, sigma_j_next, n_j_next):
		self.mu_j = Variable(mu_j)
		self.sigma_j = Variable(sigma_j)
		self.n_j = Variable(n_j)
		self.mu_j_next = Variable(mu_j_next)
		self.sigma_j_next = Variable(sigma_j_next)
		self.n_j_next = Variable(n_j_next)

	def forward(self, input):
		assert len(input.shape) in (2, 3, 4)
		if input.device.type != "cpu":
			self.mu_j = self.mu_j.cuda()
			self.sigma_j = self.sigma_j.cuda()
			self.n_j = self.n_j.cuda()
		output, next  = adaptive_batch_norm(input, self.weight, self.bias, self.mu_j, self.sigma_j, self.n_j, is_training = self.training, eps = self.eps)
		self.update_next_stats(next)

		return output

	def update_next_stats(self, next):
		assert len(next) == 3
		self.mu_j_next = Variable(next[0])
		self.sigma_j_next = Variable(next[1])
		self.n_j_next = Variable(next[2])

	def _load_attr(self, path):
		if os.path.exists(path):
			with open(path, "rb") as f:
				# attr = pickle.load(f)
				if self.use_cuda:
					attr = torch.load(f, map_location = torch.device('cuda'))
				else:
					attr = torch.load(f, map_location = torch.device('cpu'))
		else:
			attr = None
		return attr

	def load_attrs(self):
		if os.path.exists(self.variables_dir):
			self.mu_j = self._load_attr(os.path.join(self.variables_dir, AdaBN.MU_J_FILE))
			self.sigma_j = self._load_attr(os.path.join(self.variables_dir, AdaBN.SIGMA_J_FILE))
			self.n_j = self._load_attr(os.path.join(self.variables_dir, AdaBN.N_J_FILE))
			self.mu_j_next = self._load_attr(os.path.join(self.variables_dir, AdaBN.MU_J_NEXT_FILE))
			self.sigma_j_next = self._load_attr(os.path.join(self.variables_dir, AdaBN.SIGMA_J_NEXT_FILE))
			self.n_j_next = self._load_attr(os.path.join(self.variables_dir, AdaBN.N_J_NEXT_FILE))
			if ((self.mu_j == None) or (self.sigma_j == None) or (self.n_j == None) or (self.mu_j_next == None) or (self.sigma_j_next == None) or (self.n_j_next == None)):
				self.reset_running_stats()
				print("reset running stats!")
			print("load attrs from dict successfully!")

	def _save_attr(self, path, attr):
		with open(path, "wb") as f:
			# pickle.dump(attr, f)
			torch.save(attr, f)

	def save_attrs(self):
		self._save_attr(os.path.join(self.variables_dir, AdaBN.MU_J_FILE), self.mu_j)
		self._save_attr(os.path.join(self.variables_dir, AdaBN.SIGMA_J_FILE), self.sigma_j)
		self._save_attr(os.path.join(self.variables_dir, AdaBN.N_J_FILE), self.n_j)
		self._save_attr(os.path.join(self.variables_dir, AdaBN.MU_J_NEXT_FILE), self.mu_j_next)
		self._save_attr(os.path.join(self.variables_dir, AdaBN.SIGMA_J_NEXT_FILE), self.sigma_j_next)
		self._save_attr(os.path.join(self.variables_dir, AdaBN.N_J_NEXT_FILE), self.n_j_next)
		print("save attrs into pkl successfully!")

def adaptive_batch_norm(input, gamma, beta, mu_j, sigma_j, n_j, is_training = True, eps = 1e-5, ):
	if is_training:
		_varify_batch_size(input.size())
	return _adaptive_batch_norm(input, gamma, beta, mu_j, sigma_j, n_j, is_training = is_training, eps = eps)

def _varify_batch_size(size):
	size_prods = size[0]
	for i in range(len(size) - 2):
		size_prods *= size[i + 2]
	if size_prods == 1:
		raise ValueError("Expected more than 1 vaule per channel when training, got input size {}".format(size))

def _adaptive_batch_norm(input, gamma, beta, mu_j_src, sigma_j_src, n_j_src, is_training = True, eps = 1e-5, ):
	if len(input.shape) == 2:
		# adabn_1d
		shape_1d = (1, input.shape[1])
		n_j = n_j_src.view(1)
		# finetune gamma & beta & mu_j & sigma_j
		gamma = gamma.view(shape_1d)
		beta = beta.view(shape_1d)
		mu_j = mu_j_src.view(shape_1d)
		sigma_j = sigma_j_src.view(shape_1d)
		# get curr_batch's mu & sigma
		mu = torch.mean(input, dim = 0).view(shape_1d)
		sigma = torch.mean(torch.sub(input, mu).pow(2), dim = 0).view(shape_1d)
		if is_training:
			# get constants
			k = input.shape[0]
			d = torch.sub(mu, mu_j)
			if n_j == 0:
				# after init
				mu_j = mu
			else:
				mu_j = torch.add(mu_j, torch.div(torch.mul(d, k), n_j))
			sigma_j = torch.add(torch.div(torch.mul(sigma_j, n_j), torch.add(n_j, k)), torch.add(torch.div(torch.mul(sigma, k), torch.add(n_j, k)), torch.div(torch.mul(d.pow(2), torch.mul(n_j, k)), torch.add(n_j, k).pow(2))))
			n_j = torch.add(n_j, k)
	elif len(input.shape) == 3:
		# adabn_1d
		shape_1d = (1, input.shape[1], 1)
		n_j = n_j_src.view(1)
		# finetune gamma & beta & mu_j & sigma_j
		gamma = gamma.view(shape_1d)
		beta = beta.view(shape_1d)
		mu_j = mu_j_src.view(shape_1d)
		sigma_j = sigma_j_src.view(shape_1d)
		# get curr_batch's mu & sigma
		mu = torch.mean(input, dim = (0, 2)).view(shape_1d)
		sigma = torch.mean(torch.sub(input, mu).pow(2), dim = (0, 2)).view(shape_1d)
		if is_training:
			# get constants
			k = input.shape[0]
			d = torch.sub(mu, mu_j)
			if n_j == 0:
				# after init
				mu_j = mu
			else:
				mu_j = torch.add(mu_j, torch.div(torch.mul(d, k), n_j))
			sigma_j = torch.add(torch.div(torch.mul(sigma_j, n_j), torch.add(n_j, k)), torch.add(torch.div(torch.mul(sigma, k), torch.add(n_j, k)), torch.div(torch.mul(d.pow(2), torch.mul(n_j, k)), torch.add(n_j, k).pow(2))))
			n_j = torch.add(n_j, k)
	elif len(input.shape) == 4:
		# adabn_2d
		shape_2d = (1, input.shape[1], 1, 1)
		n_j = n_j_src.view(1)
		# finetune gamma & beta & mu_j & sigma_j
		gamma = gamma.view(shape_2d)
		beta = beta.view(shape_2d)
		mu_j = mu_j_src.view(shape_2d)
		sigma_j = sigma_j_src.view(shape_2d)
		# get curr_batch's mu & sigma
		mu = torch.mean(input, dim = (0, 2, 3)).view(shape_2d)
		sigma = torch.mean(torch.sub(input, mu).pow(2), dim = (0, 2, 3)).view(shape_2d)
		if is_training:
			# get constants
			k = input.shape[0]
			d = torch.sub(mu, mu_j)
			if n_j == 0:
				# after init
				mu_j = mu
			else:
				mu_j = torch.add(mu_j, torch.div(torch.mul(d, k), n_j))
			sigma_j = torch.add(torch.div(torch.mul(sigma_j, n_j), torch.add(n_j, k)), torch.add(torch.div(torch.mul(sigma, k), torch.add(n_j, k)), torch.div(torch.mul(d.pow(2), torch.mul(n_j, k)), torch.add(n_j, k).pow(2))))
			n_j = torch.add(n_j, k)
	elif len(input.shape) == 5:
		# adabn_3d
		shape_3d = (1, input.shape[1], 1, 1, 1)
		n_j = n_j_src.view(1)
		# finetune gamma & beta & mu_j & sigma_j
		gamma = gamma.view(shape_3d)
		beta = beta.view(shape_3d)
		mu_j = mu_j_src.view(shape_3d)
		sigma_j = sigma_j_src.view(shape_3d)
		# get curr_batch's mu & sigma
		mu = torch.mean(input, dim = (0, 2, 3, 4)).view(shape_3d)
		sigma = torch.mean(torch.sub(input, mu).pow(2), dim = (0, 2, 3, 4)).view(shape_3d)
		if is_training:
			# get constants
			k = input.shape[0]
			d = torch.sub(mu, mu_j)
			if n_j == 0:
				# after init
				mu_j = mu
			else:
				mu_j = torch.add(mu_j, torch.div(torch.mul(d, k), n_j))
			sigma_j = torch.add(torch.div(torch.mul(sigma_j, n_j), torch.add(n_j, k)), torch.add(torch.div(torch.mul(sigma, k), torch.add(n_j, k)), torch.div(torch.mul(d.pow(2), torch.mul(n_j, k)), torch.add(n_j, k).pow(2))))
			n_j = torch.add(n_j, k)
	else:
		raise NotImplementedError

	output = torch.add(torch.mul(gamma, torch.div(torch.sub(input, mu_j), torch.add(sigma_j, eps).sqrt())), beta)

	return output, (mu_j.squeeze(), sigma_j.squeeze(), n_j.squeeze())

