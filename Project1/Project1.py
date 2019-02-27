
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
U,S,V = linalg.svd(Xc,full_matrices=False)
V = V.T

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()
# Project data onto principal component space usin dot product (@)
Z = Xc @ V

threshold = 0.90

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


# Plot PCA of the data
f = plt.figure()
plt.title('CHD projected on PCs')
for c in n:
    # select indices belonging to class c:
    class_mask = (y == c)
    plt.plot(Z[class_mask,0], Z[class_mask,1], 'o')
plt.legend(n)
plt.xlabel('PC1')
plt.ylabel('PC2')

#Coefficients of the PCA components per attribute
plt.figure()
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames, rotation=45)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('PCA Component Coefficients')
plt.show()


#Plot attributes' standar deviations
r = np.arange(1,X.shape[1]+1)
plt.bar(r, np.std(X,0))
plt.xticks(r, attributeNames, rotation=45)
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('Attribute standard deviations')

#Standardize X to a unit standard deviation
Xs = Xc/np.std(Xc,0)
Xcs = [Xc, Xs]

titles = ['Zero-mean', 'Zero-mean and unit variance']
threshold = 0.9
# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot
plt.figure(figsize=(10,15))
plt.subplots_adjust(hspace=.4)
plt.title('NanoNose: Effect of standardization')
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

plt.show()
