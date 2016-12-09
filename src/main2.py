import pandas as pd
from EMbayes import *
import numpy as np
from EMnaivebayes import *

datafile = '../data/car.data'
data = pd.read_csv(datafile, header=None)
K = len(set(data.iloc[:,0]))
#ans =[data.iloc[:,0].count(x) for x in set(data.iloc[:,0])]
print K
data = data.iloc[:,1:]
X  = np.array(pd.get_dummies(data))
m = EMNaiveBayes(epsilon=1e-5)
m.fit(X, K, max_iter = 100)
print m.pyk
m2 = EMbayes(epsilon = 1e-5)
m2.fit(X, K,maxiter = 100)
print m2.q
print m2.qjd
