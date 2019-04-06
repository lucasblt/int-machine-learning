from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show, savefig
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection, tree
from scipy import stats
import pandas as pd
import torch
from toolbox_02450 import rlr_validate, train_neural_net, draw_neural_net

########################### DATA LOADING ###########################
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

#Attribute to be predicted based on the other attributes
y_idx = attributeNames.index('chd')
y = X[:,y_idx]

X_cols = list(range(0,y_idx))
X = X[:,X_cols]
N, M = X.shape

#Remove y and binary attributes from atrributeNames
attributeNames = attributeNames[0:y_idx]

########################### CROSSVALIDATION ###########################
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=False)
CV2 = model_selection.KFold(n_splits=K,shuffle=False)

# Initialize variables
Error_rlr = np.empty((K,1))
Error_baseline = np.empty((K,1))
Error_ann = np.empty((K,1))
opt_lambda = np.empty((K,1))
opt_val_err = np.empty((K,1))
opt_h_val_err = np.empty((K,1))
opt_h = np.empty((K,1))
n_tested=0
lambdas = np.power(10.,range(-2,9))
hidden = np.arange(5)+1

# Outer CV
k=0

for train_index, test_index in CV.split(X,y):
    # extract training and test set for current CV fold
    X_train_out = X[train_index,:]
    y_train_out = y[train_index]
    X_test_out = X[test_index,:]
    y_test_out = y[test_index]

    # Standardize the training and set set based on training set moments
    mu = np.mean(X_train_out[:, 1:], 0)
    sigma = np.std(X_train_out[:, 1:], 0)
        
    X_train_out[:, 1:] = (X_train_out[:, 1:] - mu) / sigma
    X_test_out[:, 1:] = (X_test_out[:, 1:] - mu) / sigma

    # Inner CV
    lambda_error = np.empty((K,len(lambdas)))
    hidden_error = np.empty((K,len(hidden)))

    k2 = 0
    for train_index, test_index in CV2.split(X_train_out,y_train_out):
        print('Outer-CV-fold {0} of {1}'.format(k+1,K))
        print('Inner-CV-fold {0} of {1}'.format(k2+1,K))

        # extract training and test set for current CV fold
        X_train_in = X_train_out[train_index,:]
        y_train_in = y_train_out[train_index]
        X_test_in = X_train_out[test_index,:]
        y_test_in = y_train_out[test_index]
        
        # Extract training and test set for current CV fold, 
        # and convert them to PyTorch tensors
        X_train2 = torch.tensor(X_train_in, dtype=torch.float)
        y_train2 = torch.tensor(y_train_in, dtype=torch.float)
        X_test2 = torch.tensor(X_test_in, dtype=torch.float)
        y_test2 = torch.tensor(y_test_in, dtype=torch.uint8)
        
        for l in range(0,len(lambdas)):
            clf1 = lm.RidgeClassifier(alpha=lambdas[l]).fit(X_train_in, y_train_in)
            y_logreg = clf1.predict(X_test_in)
            lambda_error[k2,l] = 100*(y_logreg!=y_test_in).sum().astype(float)/len(y_test_in)
        
        for h in range(0,len(hidden)):
            n_hidden_units = hidden[h]
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units),
            
                                torch.nn.Tanh(),   
                                torch.nn.Linear(n_hidden_units, 1),
                                torch.nn.Sigmoid()
                                )
            loss_fn = torch.nn.BCELoss()
            max_iter = 10000
            net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train2,
                                                       y=y_train2,
                                                       n_replicates=1,
                                                       max_iter=max_iter)
            
            # Determine estimated class labels for test set
            y_sigmoid = net(X_test2) # activation of final note, i.e. prediction of network
            y_test_est = y_sigmoid > .5 # threshold output of sigmoidal function
            
            # Determine errors and error rate
            e = (y_test_est != y_test2)[0]
            hidden_error[k2,h] = 100*(sum(e).type(torch.float)/len(y_test_in)).data.numpy()
        k2 += 1
    
    ######### Selecting Lambda for the RLR ##########
    opt_val_err[k] = np.min(np.mean(lambda_error,axis=0))
    opt_lambda[k] = lambdas[np.argmin(np.mean(lambda_error,axis=0))]
    
    ######### Selecting hidden units for the ANN ##########
    opt_h_val_err[k] = np.min(np.mean(hidden_error,axis=0))
    opt_h[k] = hidden[np.argmin(np.mean(hidden_error,axis=0))]    
    
    # Evaluate rlr model
    Error_rlr[k] = opt_val_err[k]
    
    # Evaluate ann model
    Error_ann[k] = opt_h_val_err[k]
    
    # Evaluate baseline model
    pos = y_test_out.sum()/len(y_test_out)
    if pos > 0.5:
        y_base = np.ones(len(y_test_out))
    else:
        y_base = np.zeros(len(y_test_out))
        
    Error_baseline[k] = 100*(y_base!=y_test_out).sum().astype(float)/len(y_test_out)

    k += 1

########################### COMPARISONS ###########################

# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
# [tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_dectree)
# and test if the p-value is less than alpha=0.05.
    
## BASELINE VS RLR ##

z = (Error_baseline-Error_rlr)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL1 = zb + sig * stats.t.ppf(alpha/2, nu);
zH1 = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL1 <= 0 and zH1 >= 0 :
    print('Models are not significantly different')        
else:
    print('Models are significantly different.')
    
# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_baseline, Error_rlr),axis=1))
xlabel('Baseline   vs.   Regularized Logistic Regression')
ylabel('Cross-validation error [%]')
plt.savefig('figures/classification/baseline_logistic.png',bbox_inches = 'tight')
show()

## BASELINE VS ANN ##

z = (Error_baseline-Error_ann)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL2 = zb + sig * stats.t.ppf(alpha/2, nu);
zH2 = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL2 <= 0 and zH2 >= 0 :
    print('Models are not significantly different')        
else:
    print('Models are significantly different.')
    
# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_baseline, Error_ann),axis=1))
xlabel('Baseline   vs.   ANN')
ylabel('Cross-validation error [%]')
savefig('figures/classification/baseline_ann.png',bbox_inches = 'tight')
show()

## RLR VS ANN ##

z = (Error_rlr-Error_ann)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL3 = zb + sig * stats.t.ppf(alpha/2, nu);
zH3 = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL3 <= 0 and zH3 >= 0 :
    print('Models are not significantly different')        
else:
    print('Models are significantly different.')
    
# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_rlr, Error_ann),axis=1))
xlabel('Regularized Logistic Regression   vs.   ANN')
ylabel('Cross-validation error [%]')
savefig('figures/classification/logistic_ann.png',bbox_inches = 'tight')

show()
