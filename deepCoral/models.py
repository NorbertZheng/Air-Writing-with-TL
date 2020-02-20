import torch.nn as nn
# local model
import sys
sys.path.append("../network")
from Coral import CORAL
import mmd
import backbone

class Transfer_Net(nn.Module):

	def __init__(self, n_class, base_net = "naive_cnnblstm", transfer_loss = "mmd", use_bottleneck = True, bottleneck_width = 256, width = 1024, use_cuda = False):
		super(Transfer_Net, self).__init__()
		self.base_network = backbone.network_dict[base_net](use_cuda = use_cuda)
		
