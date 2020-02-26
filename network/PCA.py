import numpy as np

class PCA:
	def __init__(self, target_dimension = 3):
		self.target_dimension = target_dimension

	def process(self, features):
		self.features = features
		dots = self.features

		cov = np.cov(np.matrix(dots).T)

		eigenvalue, feature_vector = np.linalg.eig(cov)

		reduce_matrix = feature_vector[np.argsort(-eigenvalue)[:self.target_dimension]]
		print('%s dimension(s) reduce_matrix.' % self.target_dimension)
		print(reduce_matrix)

		return np.matrix(self.features) * reduce_matrix.T
