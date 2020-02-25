import numpy as np
from sklearn import linear_model

class SCL(object):
	'''
	class of structural correspondence learning 
	'''
	def __init__(self, l2=1.0, num_pivots=10):
		self.l2 = l2
		self.num_pivots = num_pivots
		self.W = 0
		# self.train_data_dim = None

	def fit(self, Xs, Xt):
		'''
		find pivot features and transfer the Xs and Xt
		Param Xs: source data
		Param Xt: target data
		output Xs_new: new source data features
		output Xt_new: new target data features
		output W: transform matrix
		'''
		_, ds = Xs.shape
		_, dt = Xt.shape
		assert ds == dt
		X = np.concatenate((Xs, Xt), axis=0)
		ix = np.argsort(np.sum(X, axis=0))
		ix = ix[::-1][:self.num_pivots]
		pivots = (X[:, ix]>0).astype('float')
		p = np.zeros((ds, self.num_pivots))
		# train for the classifers 
		for i in range(self.num_pivots):
			clf = linear_model.SGDClassifier(loss="modified_huber", alpha=self.l2)
			clf.fit(X, pivots[:, i])
			p[:, i] = clf.coef_
		_, W = np.linalg.eig(np.cov(p))
		W = W[:, :self.num_pivots].astype('float')
		self.W = W
		Xs_new = np.concatenate((np.dot(Xs, W), Xs), axis=1)
		Xt_new = np.concatenate((np.dot(Xt, W), Xt), axis=1)

		return Xs_new, Xt_new, W

	def transform(self, X):
		'''
		transform the origianl data by add new features
		Param X: original data
		output x_new: X with new features
		'''
		X_new = np.concatenate((np.dot(X, self.W),X), axis=1)
		return X_new

if __name__ == "__main__":
	Xs = np.random.rand(2, 5)
	Xt = np.random.rand(2, 5)
	scl = SCL()
	Xs_new, Xt_new, _ = scl.fit(Xs, Xt)
