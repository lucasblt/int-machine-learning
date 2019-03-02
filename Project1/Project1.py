
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import numpy as np
import pandas as pd

# Load the csv data using the Pandas library
filename = 'https://web.stanford.edu/~hastie/ElemStatLearn//datasets/SAheart.data'
df = pd.read_csv(filename)

raw_data = df.get_values()

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

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rhoc)+1),rhoc,'x-')
plt.plot(range(1,len(rhoc)+1),np.cumsum(rhoc),'o-')
plt.plot([1,len(rhoc)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.savefig('Figures/variance_explained.png')
plt.show()


# Plot PCA of the data
f = plt.figure()
plt.title('CHD projected on PCs')
for c in n:
    # select indices belonging to class c:
    class_mask = (y == c)
    plt.plot(Zc[class_mask,0], Zc[class_mask,1], 'o',alpha=0.5)
plt.legend(n)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('Figures/projectedPCA.png')

#Coefficients of the PCA components per attribute
plt.figure()
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
plt.title('PCA Component Coefficients')
plt.savefig('Figures/attributes_PCA.png')
plt.show()


#Plot attributes' standar deviations
r = np.arange(1,X.shape[1]+1)
plt.bar(r, np.std(X,0))
plt.xticks(r, attributeNames, rotation=45)
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('Attribute standard deviations')
plt.savefig('Figures/attributes_std.png')

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
    plt.xlabel('PC'+str(j+1))
    plt.title(titles[k] + '\n' + 'Projection' )
    plt.legend(n)
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

plt.savefig('Figures/principal_components.png')
plt.show()

#Summary statistics
X_mean = X.mean(axis=0)
X_std = X.std(axis=0,ddof=1)
X_median = np.median(X,axis=0)
X_range = X.max(axis=0)-X.min(axis=0)

cov_X = np.cov(X, rowvar=False, ddof=1)
corrcoef_X = np.corrcoef(X, rowvar=False)


plt.figure(figsize=(8,10))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    if i != 4:
        plt.subplot(u,v,i+1)
        plt.hist(X[:,i], color=(0.2, 0.8-i*0.1, 0.4))
        plt.xlabel(attributeNames[i])
        plt.ylim(0,N/2)
    
plt.savefig('Figures/attributes_histogram.png')
plt.show()
#Boxplot of each atribbute of the dataset
#without the famhist attribute
plt.figure(figsize=(8,10))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    if i != 4:
        plt.subplot(u,v,i+1)
        plt.boxplot(X[:,i])
        plt.xticks([],attributeNames[i])
        plt.ylabel('')
        plt.title(attributeNames[i])
plt.savefig('Figures/attributes_boxplot.png')
plt.show()

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

plt.savefig('Figures/class_boxplot.png')

plt.show()


Xaux = np.hstack((Xs[:,0:4],Xs[:,5:]))
plt.figure(figsize=(24,20))

for m1 in range(M-1):
    for m2 in range(M-1):
        plt.subplot(M-1, M-1, m1*(M-1) + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plt.plot(np.array(Xaux[class_mask,m2]), np.array(Xaux[class_mask,m1]), '.',alpha=0.5)
            if m1==M-2:
                plt.xlabel(attributeNames[m2])
            else:
                plt.xticks([])
            if m2==0:
                plt.ylabel(attributeNames[m1])
            else:
                plt.yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
plt.legend(classNames)
plt.savefig('Figures/attribute_correlation.png')

plt.show()

#Statistics
stats = np.vstack((attributeNames,X.mean(0), X.std(0),X.var(0))).T
stat_df = pd.DataFrame(stats)
stat_df.to_excel('attibutesStats.xlsx',index=False)

PCA = np.array([['PC{}'.format(i+1) for i in range(len(rho))]]).T
#Principal components Centralized and Standardized
pc_c = np.vstack((np.around(rhoc,decimals=4),np.around(np.cumsum(100*rhoc),decimals=4))).T
pc_s = np.vstack((np.around(rhos,decimals=4),np.around(np.cumsum(100*rhos),decimals=4))).T

pc_df = pd.DataFrame(np.hstack((PCA,pc_c,pc_s)))
pc_df.to_excel('PCcomponents.xlsx',index=False)
