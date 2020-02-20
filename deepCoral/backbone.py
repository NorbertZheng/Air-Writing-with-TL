import torch.nn as nn
# local model
import sys
sys.path.append("../cnnblstm")
from cnnblstm import cnnblstm

class naive_cnnblstm(nn.Module):
	PRE_TRAINED = True
	PARAMS_FILE = "./params.pkl"
	N_CLASSIFIER = 2

	def __init__(self, time_steps = 800, n_features = 3, n_outputs = 10, params_file = naive_cnnblstm.PARAMS_FILE, use_cuda = False):
		# get naive cnnblstm
		m_cnnblstm = cnnblstm(time_steps = time_steps, n_features = n_features, n_outputs = n_outputs, params_file = params_file, use_cuda = use_cuda).get_model(pre_trained = naive_cnnblstm.PRE_TRAINED)
		# set self.net
		self.net = m_cnnblstm
		self._in_features = m_cnnblstm.n_outputs

	def forward(self, input):
		output = self.net(input)
		return output

	def n_output(self):
		return self._in_features

network_dict = {"naive_cnnblstm": naive_cnnblstm}

		
