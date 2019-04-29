from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid, savefig)
import numpy as np
import pandas as pd
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Load the csv dataset into a pandas data frame
filename = 'https://web.stanford.edu/~hastie/ElemStatLearn//datasets/SAheart.data'
df = pd.read_csv(filename)
raw_data = df.get_values()

X = raw_data[:,1:]
N,M = X.shape
y = X[:,-2]

cols = range(1, M+1)

#Extract attribute names
attributeNames = np.asarray(df.columns[cols]).tolist()

X_cols = [0,1,2,3,6,7]
X = X[:,X_cols]

X = np.array(X, dtype=np.float32)

mu = np.mean(X, 0)
sigma = np.std(X, 0)
        
X = (X - mu) / sigma

### Gausian Kernel density estimator
# cross-validate kernel width by leave-one-out-cross-validation
# (efficient implementation in gausKernelDensity function)
# evaluate for range of kernel widths
widths = X.var(axis=0).max() * (2.0**np.arange(-10,3))
logP = np.zeros(np.size(widths))
for i,w in enumerate(widths):
   print('Fold {:2d}, w={:f}'.format(i,w))
   density, log_density = gausKernelDensity(X,w)
   logP[i] = log_density.sum()
   
val = logP.max()
ind = logP.argmax()

width=widths[ind]
print('Optimal estimated width is: {0}'.format(width))

# evaluate density for estimated width
density, log_density = gausKernelDensity(X,width)

# Sort the densities
i = (density.argsort(axis=0)).ravel()
density = density[i].reshape(-1,)

# Plot density estimate of outlier score
figure(figsize=(14,9)); 
plt.bar(range(100),density[:100])
title('Density estimate')
savefig('figures/OUTLIER/gaussian.png',bbox_inches = 'tight')


### K-neighbors density estimator
# Neighbor to use:
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

density = 1./(D.sum(axis=1)/K)

# Sort the scores
i = density.argsort()
density = density[i]

# Plot k-neighbor estimate of outlier score (distances)
figure(figsize=(14,9)); 
plt.bar(range(100),density[:100])
title('KNN density: Outlier score')
savefig('figures/OUTLIER/KNNdensity.png',bbox_inches = 'tight')

### K-nearest neigbor average relative density
# Compute the average relative density

knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)
density = 1./(D.sum(axis=1)/K)
avg_rel_density = density/(density[i[:,1:]].sum(axis=1)/K)

# Sort the avg.rel.densities
i_avg_rel = avg_rel_density.argsort()
avg_rel_density = avg_rel_density[i_avg_rel]

# Plot k-neighbor estimate of outlier score (distances)
figure(figsize=(14,9)); 
plt.bar(range(100),avg_rel_density[:100])
title('KNN average relative density: Outlier score')
savefig('figures/OUTLIER/KNNrelativedensity.png',bbox_inches = 'tight')
