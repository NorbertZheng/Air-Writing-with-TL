from deepAE import deepAE
from ConvAE import ConvAE
from SeqAE import SeqAE

AutoEncoder = {"deepAE": deepAE, "ConvAE": ConvAE, "SeqAE": SeqAE}

def load_AE(type = "ConvAE", time_steps = 800, n_features = 3, use_cuda = False, params_pkl = "./params_ConvAE.pkl"):
	m_ae = AutoEncoder[type](time_steps = time_steps, n_features = n_features, use_cuda = use_cuda, params_pkl = params_pkl):
	return m_ae

def train_AE(m_ae, src_x, target_x):
	return m_ae.get_model(src_x = src_x, target_x = target_x, pre_trained = False)
