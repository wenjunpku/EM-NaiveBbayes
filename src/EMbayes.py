import numpy as np
import pandas as  pd
import math
import copy
class EMbayes(object):

	def __init__(self, epsilon):
		self.epsilon = epsilon

	def _init_params(self, X, K, maxiter):
		self.X = X
		self.K = K
		self.M, self.N = X.shape
		self.maxiter = maxiter
		self.delta =np.tile(np.array([0.0 for i in range(self.M)]),(self.K, 1))

		self.q = np.random.rand(self.K)
		tmpsum = self.q.sum()
		self.q = np.array([x/tmpsum for x in self.q])
		self.q_old = copy.deepcopy(self.q)

		self.qj =np.array([np.random.rand(self.K), np.random.rand(self.K)])
		self.qj = self.qj/self.qj.sum(axis = 0)
		self.qjd = []
		for i in range(self.N):
			tmp = np.array([np.random.rand(self.K), np.random.rand(self.K)])
			tmp = tmp/tmp.sum(axis = 0)
			self.qjd.append(tmp)
		self.qjd = np.array(self.qjd)


	def _iterate(self):
		self.q_old = copy.deepcopy(self.q)

		for i in range(self.M):
			ssum = 0.0;
			for y in range(self.K):
				tmp = 0.0
				for j in range(self.N):
					if(self.qjd[j][1 if self.X[i][j] == 1.0 else 0][y] < 0.0):
						print "num1 errer"
					if(self.qjd[j][1 if self.X[i][j] == 1.0 else 0][y] == 0.0):
						print "num2 errer"
					tmp += math.log(self.qjd[j][1 if self.X[i][j] == 1.0 else 0][y])
				self.delta[y][i] = self.q[y] * math.exp(tmp);
				ssum += self.delta[y][i]
			for y in range(self.K):
				self.delta[y][i] = self.delta[y][i] / ssum;

		for y in range(self.K):
			ssum2 = 0.0
			for i in range(self.M):
				ssum2 += self.delta[y][i]
			self.q[y] = ssum2/float(self.M)

			tmpsum = [0.0 for j in range(self.N)];
			for j in range(self.N):
				for i in range(self.M):
					if self.X[i][j] == 1.0:
						tmpsum[j] += self.delta[y][i]
			for j in range(self.N):
				self.qjd[j][1][y] = tmpsum[j]/ssum2;
				self.qjd[j][0][y] = 1.0 - self.qjd[j][1][y]
				if self.qjd[j][0][y] == 0.0:
					self.qjd[j][0][y] =1e-6
					self.qjd[j][1][y] = 1.0 - self.qjd[j][0][y]
				if self.qjd[j][1][y] == 0.0:
					self.qjd[j][1][y] = 1e-6;
					self.qjd[j][0][y] = 1.0 - self.qjd[j][1][y]

		return self
	def _is_convergent(self):
		diff = 0.0
		for q_old, q_new in zip(self.q_old, self.q):
			diff += abs(q_old - q_new)
		return diff < self.epsilon

	def fit(self, X, K , maxiter = 100):
		self._init_params(X, K, maxiter)
		for i in range(maxiter):
			print i
			print self.q
			self._iterate()
			if self._is_convergent():
				break;
		return self

if __name__ == '__main__':
	X, K = np.array([[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]),2
	m = EMbayes(epsilon=1e-5)
	m.fit(X, K, maxiter = 10)
	print m.q
	print m.qjd
