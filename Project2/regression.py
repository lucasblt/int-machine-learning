from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid, savefig)
import scipy.linalg as linalg
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate, train_neural_net, draw_neural_net
import matplotlib.pyplot as plt
import torch

# Load the csv dataset into a pandas data frame
filename = 'https://web.stanford.edu/~hastie/ElemStatLearn//datasets/SAheart.data'
df = pd.read_csv(filename)
raw_data = df.get_values()

X = raw_data[:,1:]
N,M = X.shape
y = np.zeros((5,1))

cols = range(1, M+1)

#Extract attribute names
attributeNames = np.asarray(df.columns[cols]).tolist()

#Convert attribute 'famhist' with list comprehension: 0-Absent; 1-Present 
famHistNames = X[:,4]
famHistLabels = np.unique(famHistNames)
famHistDict = dict(zip(famHistLabels,[0,1]))
famHistLc = np.array([famHistDict[cl] for cl in famHistNames])
X = np.hstack((X[:,0:4],np.array([famHistLc]).T,X[:,5:])).astype(np.float)

#Binary attributes to be removed from dataset
famhist_idx = attributeNames.index('famhist')
chd_idx = attributeNames.index('chd')

#Attribute to be predicted based on the other attributes
y_idx = attributeNames.index('ldl')
y = X[:,y_idx]

X_cols = list(range(0,y_idx))+ list(range(y_idx+1,famhist_idx))+list(range(famhist_idx+1,chd_idx))
X = X[:,X_cols]
N, M = X.shape
#Remove y and binary attributes from atrributeNames
attributeNames = attributeNames[0:y_idx] + attributeNames[y_idx+1:famhist_idx] + attributeNames[famhist_idx+1:chd_idx]

########################## REGRESSION - PART A ##########################
# 1) REGULARIZATION PARAMETER
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10

# Values of lambda
lambdas = np.power(10.,range(-2,9))

# Initialize variables
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))


CV = model_selection.KFold(K, shuffle=False)
w = np.empty((M,K,len(lambdas)))
train_error = np.empty((K,len(lambdas)))
test_error = np.empty((K,len(lambdas)))
k = 0
y = y.squeeze()

for train_index, test_index in CV.split(X,y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
        
    # Standardize the training and set set based on training set moments
    mu = np.mean(X_train[:, 1:], 0)
    sigma = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
    X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
    
    # precompute terms
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    for l in range(0,len(lambdas)):
        # Compute parameters for current value of lambda and current CV fold
        lambdaI = lambdas[l] * np.eye(M)
        lambdaI[0,0] = 0 # remove bias regularization
        w[:,k,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Evaluate training and test performance
        train_error[k,l] = np.power(y_train-X_train @ w[:,k,l].T,2).mean(axis=0)
        test_error[k,l] = np.power(y_test-X_test @ w[:,k,l].T,2).mean(axis=0)

    k=k+1
    
opt_val_err = np.min(np.mean(test_error,axis=0))
opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
train_err_vs_lambda = np.mean(train_error,axis=0)
test_err_vs_lambda = np.mean(test_error,axis=0)
mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))


figure(figsize=(12,8))

semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
legend(attributeNames[1:], loc='best')
savefig('figures/regression/part_a1.png',bbox_inches = 'tight')
     
figure(figsize=(12,8))

title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()
savefig('figures/regression/part_a2.png',bbox_inches = 'tight')
show()

for i in range(len(lambdas)):
    if lambdas[i] == opt_lambda:
        opt_lambda_idx = i
        
# Display results
print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w[m,-1,opt_lambda_idx],2)))