from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid, savefig)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolbox_02450 import clusterplot
from sklearn.cluster import k_means
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

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

#Convert attribute 'famhist' with list comprehension: 0-Absent; 1-Present 
famHistNames = X[:,4]
famHistLabels = np.unique(famHistNames)
famHistDict = dict(zip(famHistLabels,[0,1]))
famHistLc = np.array([famHistDict[cl] for cl in famHistNames])
X = np.hstack((X[:,0:4],np.array([famHistLc]).T,X[:,5:])).astype(np.float)

#X_cols = [0,1,2,3,6,7]
X_cols = [3,6]
X = X[:,X_cols]

# GMM CROSS VALIDATION ##########################################################################

# Range of K's to try
KRange = range(1,11)
T = len(KRange)

covar_type = 'full'       # you can try out 'diag' as well
reps = 10                  # number of fits with different initalizations, best result will be kept
init_procedure = 'kmeans' # 'kmeans' or 'random'

# Allocate variables
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10,shuffle=False)

for t,K in enumerate(KRange):
        print('Fitting model for K={0}'.format(K))

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, 
                              n_init=reps, init_params=init_procedure,
                              tol=1e-6, reg_covar=1e-6).fit(X)

        # For each crossvalidation fold
        for train_index, test_index in CV.split(X):

            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

            # compute negative log likelihood of X_test
            CVE[t] += -gmm.score_samples(X_test).sum()
            

# Plot results
figure(figsize=(14,9)); 
plt.plot(KRange, 2*CVE,'-ok')
legend(['Crossvalidation'])
xlabel('K')
savefig('figures/GMM/clustering_CV.png',bbox_inches = 'tight')
show()

# GMM 1 ##########################################################################
# Number of clusters
K = np.argmin(CVE) + 1

class_max = np.amax(y)
class_min = np.amin(y)

delta = (class_max - class_min)/K

y2 = np.zeros((N,))
for i in range(N):
    for j in range(K):
        if y[i] >= class_min + j*delta and y[i] < class_min + (j+1)*delta:
            y2[i] = j
            break

cov_type = 'full' # e.g. 'full' or 'diag'

# define the initialization procedure (initial value of means)
initialization_method = 'kmeans'#  'random' or 'kmeans'
# random signifies random initiation, kmeans means we run a K-means and use the
# result as the starting point. K-means might converge faster/better than  
# random, but might also cause the algorithm to be stuck in a poor local minimum 

# type of covariance, you can try out 'diag' as well
reps = 1
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps, 
                      tol=1e-6, reg_covar=1e-6, init_params=initialization_method).fit(X)
cls = gmm.predict(X)    
# extract cluster labels
cds = gmm.means_        
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)
if cov_type.lower() == 'diag':
    new_covs = np.zeros([K,M,M])    
    
    count = 0    
    for elem in covs:
        temp_m = np.zeros([M,M])
        new_covs[count] = np.diag(elem)
        count += 1

    covs = new_covs

## In case the number of features != 2, then a subset of features most be plotted instead.
figure(figsize=(14,9))
#idx = [3,4] # feature index, choose two features to use as x and y axis in the plot
#clusterplot(X[:,idx], clusterid=cls, centroids=cds[:,idx], y=y2, covars=covs[:,idx,:][:,:,idx])
clusterplot(X, clusterid=cls, centroids=cds, y=y2, covars=covs)
savefig('figures/GMM/clustering_GMM.png',bbox_inches = 'tight')
show()

# CLUSTERING 2 ########################################################################## Perform hierarchical/agglomerative clustering on data matrix
#Method = 'single'
#Method = 'complete'
#Method = 'average'
Method = 'weighted'
#Method = 'centroid'
#Method = 'median'
#Method = 'ward'

Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = K
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
figure(figsize=(14,9))
#clusterplot(X[:,idx], cls.reshape(cls.shape[0],1), y=y2)
clusterplot(X, cls.reshape(cls.shape[0],1), y=y2)
savefig('figures/GMM/clustering_hierarchical.png',bbox_inches = 'tight')

# Display dendrogram
max_display_levels=6
figure(figsize=(14,9)); 
dendrogram(Z, truncate_mode='level', p=max_display_levels)
savefig('figures/GMM/clustering_dendrogram.png',bbox_inches = 'tight')

show()
