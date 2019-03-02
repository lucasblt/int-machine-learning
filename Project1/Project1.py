#02450 - Introduction to Machine Learning and Data Mining
#Assignment 1
#Students:
#Everton Spader - s190045
#Lucas Beltram - s182360

import matplotlib.pyplot as plt
import scipy.linalg as linalg
import numpy as np
import pandas as pd

# Load the csv dataset into a pandas data frame
filename = 'https://web.stanford.edu/~hastie/ElemStatLearn//datasets/SAheart.data'
df = pd.read_csv(filename)
raw_data = df.get_values()

#
X = raw_data[:,1:-1]
y = raw_data[:,-1]

N,M = X.shape
n = np.unique(y)
classNames = ['CHD Negative','CHD Positive']
C = len(classNames)

cols = range(1, M+1) 

#Convert attribute 'famhist' with list comprehension
attributeNames = np.asarray(df.columns[cols])
 
famHistNames = X[:,4]
famHistLabels = np.unique(famHistNames)

famHistDict = dict(zip(famHistLabels,[0,1]))

famHistLc = np.array([famHistDict[cl] for cl in famHistNames])

#Insert the list comprehension for column famhist in the raw_data
X = np.hstack((X[:,0:4],np.array([famHistLc]).T,X[:,5:])).astype(np.float)
# Center the data (subtract mean column values)
Xc = X - np.ones((N,1))*X.mean(0)
# PCA by computing SVD of Y
Uc,Sc,Vc = linalg.svd(Xc,full_matrices=False)
Vc = Vc.T
# Compute variance explained by principal components
rhoc = (Sc*Sc) / (Sc*Sc).sum()
# Project data onto principal component space usin dot product (@)
Zc = Xc @ Vc


#Standardize and compute PCA components
Xs = Xc/np.std(Xc,0)
Us,Ss,Vs = linalg.svd(Xs,full_matrices=False)
Vs = Vs.T
rhos = (Ss*Ss) / (Ss*Ss).sum()

Zs = Xs @ Vs

threshold = 0.90

fig_no = 1
#Coefficients of the PCA components per attribute
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, Vc[:,i], width=bw)
plt.xticks(r+bw, attributeNames, rotation=45)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title(r'PC of Attributes with $\mu = 0$')

plt.subplot(1,2,2)
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, Vs[:,i], width=bw)
plt.xticks(r+bw, attributeNames, rotation=45)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title(r'PC of Attributes with $\mu = 0$ and $\sigma = 1$')

plt.savefig('Figures/Fig{}_attributes_PCA.png'.format(fig_no),bbox_inches = 'tight')
plt.show()
fig_no += 1

#Plot attributes' standard deviations
r = np.arange(1,X.shape[1]+1)
plt.bar(r, np.std(X,0))
plt.xticks(r, attributeNames, rotation=45)
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('Attribute standard deviations')
plt.savefig('Figures/Fig{}_attributes_std.png'.format(fig_no),bbox_inches = 'tight')
fig_no += 1

#Standardize X to a unit standard deviation
Xcs = [Xc, Xs]

titles = ['Zero-mean', 'Zero-mean and unit variance']
threshold = 0.9
# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot
plt.figure(figsize=(10,15))
plt.subplots_adjust(hspace=.4)
nrows=3
ncols=2

for k in range(2):
    # Obtain the PCA solution by calculate the SVD of either Xc or Xs
    U,S,V = linalg.svd(Xcs[k],full_matrices=False)
    V=V.T # For the direction of V to fit the convention in the course we transpose
    # For visualization purposes, we flip the directionality of the
    # principal directions such that the directions match for Xc and Xs.
    if k==1: V = -V; U = -U; 
    
    # Compute variance explained
    rho = (S*S) / (S*S).sum() 
    
    # Compute the projection onto the principal components
    Z = U*S;
    
    # Plot projection
    plt.subplot(nrows, ncols, 1+k)
    
    C = len(n)
    for c in range(C):
        plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))
    plt.title(titles[k] + '\n' + 'Projection' )
    plt.legend(classNames)
    plt.axis('equal')
    
    # Plot attribute coefficients in principal component space
    plt.subplot(nrows, ncols,  3+k)
    for att in range(V.shape[1]):
        plt.arrow(0,0, V[att,i], V[att,j])
        plt.text(V[att,i], V[att,j], attributeNames[att])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))
    plt.grid()
    # Add a unit circle
    plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
         np.sin(np.arange(0, 2*np.pi, 0.01)));
    plt.title(titles[k] +'\n'+'Attribute coefficients')
    plt.axis('equal')
            
    # Plot cumulative variance explained
    plt.subplot(nrows, ncols,  5+k);
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.title(titles[k]+'\n'+'Variance explained')
    

plt.savefig('Figures/Fig{}_principal_components.png'.format(fig_no),bbox_inches = 'tight')
plt.show()
fig_no += 1

#Summary statistics
X_mean = X.mean(axis=0)
X_std = X.std(axis=0,ddof=1)
X_var = X.var(axis=0,ddof=1)
X_median = np.median(X,axis=0)
X_max = X.max(axis=0)
X_min = X.min(axis=0)
X_range = X_max - X_min

cov_X = np.cov(X, rowvar=False, ddof=1)
corrcoef_X = np.corrcoef(X, rowvar=False)
cov_Xs = np.cov(Xs, rowvar=False, ddof=1)
corrcoef_Xs = np.corrcoef(Xs, rowvar=False)

plt.figure(figsize=(15,6))
u = np.floor(np.sqrt(M)-1); v = np.ceil(float(M)/u)
for i in range(M):
    if i != 4:
        plt.subplot(u,v,i+1)
        plt.hist(X[:,i], color=(0.2, 0.8-i*0.1, 0.4))
        plt.xlabel(attributeNames[i])
        plt.ylim(0,N/2)


plt.savefig('Figures/Fig{}_attributes_histogram.png'.format(fig_no),bbox_inches = 'tight')
plt.show()
fig_no += 1

#Boxplot of each atribbute of the dataset
#without the famhist attribute
plt.figure(figsize=(15,6))
u = np.floor(np.sqrt(M)-1); v = np.ceil(float(M)/u)
for i in range(M):
    if i != 4:
        plt.subplot(u,v,i+1)
        plt.boxplot(X[:,i])
        plt.xticks([],attributeNames[i])
        plt.ylabel('')
        plt.title(attributeNames[i])
plt.savefig('Figures/Fig{}_attributes_boxplot.png'.format(fig_no),bbox_inches = 'tight')
plt.show()
fig_no += 1

#Plot a boxplot of each attribute for each class considering the standardized dataset
#without the famhist attribute
plt.figure(figsize=(14,7))
for c in range(C):
    plt.subplot(1,C,c+1)
    class_mask = (y==c) 
    
    plt.boxplot(np.hstack((Xs[class_mask,0:4],Xs[class_mask,5:])))
    plt.title('Class: '+classNames[c])
    plt.xticks(range(1,len(attributeNames)), [a for a in np.hstack((attributeNames[:4],attributeNames[5:]))], rotation=45)
    y_up = Xs.max()+(Xs.max()-Xs.min())*0.1; y_down = Xs.min()-(Xs.max()-Xs.min())*0.1
    plt.ylim(y_down, y_up)

plt.savefig('Figures/Fig{}_class_boxplot.png'.format(fig_no),bbox_inches = 'tight')
plt.show()
fig_no += 1

Xaux = np.hstack((Xs[:,0:4],Xs[:,5:]))
an = np.hstack((attributeNames[0:4],attributeNames[5:]))
plt.figure(figsize=(24,20))

for m1 in range(M-1):
    for m2 in range(M-1):
        plt.subplot(M-1, M-1, m1*(M-1) + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plt.plot(np.array(Xaux[class_mask,m2]), np.array(Xaux[class_mask,m1]), '.',alpha=0.5)
            if m1==M-2:
                plt.xlabel(an[m2],fontsize = 'xx-large')
            else:
                plt.xticks([])
            if m2==0:
                plt.ylabel(an[m1],fontsize = 'xx-large')
            else:
                plt.yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
plt.legend(classNames)
plt.savefig('Figures/Fig{}_attribute_correlation.png'.format(fig_no),bbox_inches = 'tight')
plt.show()

#Statistics
stats = np.vstack((attributeNames,X_mean, X_std, X_var, X_median, X_max, X_min, X_range)).T
stat_df = pd.DataFrame(stats)
stat_df.to_excel('attibutesStats.xlsx',index=False)

PCA = np.array([['PC{}'.format(i+1) for i in range(len(rho))]]).T
#Principal components Centralized and Standardized
pc_c = np.vstack((np.around(rhoc,decimals=4),np.around(np.cumsum(100*rhoc),decimals=4))).T
pc_s = np.vstack((np.around(rhos,decimals=4),np.around(np.cumsum(100*rhos),decimals=4))).T

pc_df = pd.DataFrame(np.hstack((PCA,pc_c,pc_s)))
pc_df.to_excel('PCcomponents.xlsx',index=False)

cov_matrix = np.vstack((attributeNames,cov_Xs)).T
cov_df = pd.DataFrame(cov_matrix)

corr_matrix = np.vstack((attributeNames,corrcoef_Xs)).T
corr_df = pd.DataFrame(corr_matrix)
corr_df.to_excel('corr_matrix.xlsx',index=False)

