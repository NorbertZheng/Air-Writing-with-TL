import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def CORAL(source, target):
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

