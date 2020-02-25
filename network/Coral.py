import torch
import scipy.linalg
import numpy as np
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def CORAL_loss(source, target):
	d = source.size(1)
	n_s, n_t = source.size(0), target.size(0)

	# source covariance
	tmp_s = torch.ones((1, n_s)).to(DEVICE) @ source
	cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / n_s) / (n_s - 1)

	# target covariance
	tmp_t = torch.ones((1, n_t)).to(DEVICE) @ target
	ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / n_t) / (n_t - 1)

	# frobenius norm
	loss = (cs - ct).pow(2).sum().sqrt()
	loss = loss / (4 * d * d)

	return loss

def CORAL_np(Xs, Xt):
	'''
	Perform CORAL on the source domain features
	:param Xs: ns * n_feature, source feature
	:param Xt: nt * n_feature, target feature
	:return: New source domain features
	'''
	# get domain covariance
	cov_src = np.cov(Xs.T) + 2 * np.eye(Xs.shape[1])
	cov_tar = np.cov(Xt.T) + 2 * np.eye(Xt.shape[1])

	# compute Xs_new
	A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5), scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
	Xs_new = np.dot(Xs, A_coral)

	return Xs_new

def CORAL_torch(source, target):
	return source

	n_s, n_t = source.size(0), target.size(0)

	# source covariance + source(target.size(1))
	tmp_s = torch.ones((1, n_s)).to(DEVICE) @ source
	cs = ((source.t() @ source - (tmp_s.t() @ tmp_s) / n_s) / (n_s - 1)) + torch.eye(source.size(1))

	# target covariance + eye(target.size(1))
	tmp_t = torch.ones((1, n_t)).to(DEVICE) @ target
	ct = ((target.t() @ target - (tmp_t.t() @ tmp_t) / n_t) / (n_t - 1)) + torch.eye(target.size(1))

	# TODO
	A_CORAL = source.mm()

	return None

