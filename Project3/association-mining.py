import pandas as pd
import numpy as np
from apyori import apriori

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

columns = [0,1,2,3,6,7]
X_aux = X[:,columns]

minvalues = np.amin(X_aux, axis = 0)
maxvalues = np.amax(X_aux, axis = 0)
medvalues = (minvalues + maxvalues)/2
#medvalues = np.mean(X_aux, axis = 0)

age_bin = np.zeros([N,3])

X_bin = np.zeros([N,12])

for j in range(6):
    for i in range(N):
        if X_aux[i,j] < medvalues[j]:
            X_bin[i,2*j] = 1
        elif X_aux[i,j] > medvalues[j]:
            X_bin[i,2*j+1] = 1

#AGE
for i in range(N):
    if X[i,-2] >= 15 and X[i,-2] < 30:
        age_bin[i,0] = 1
    elif X[i,-2] >= 30 and X[i,-2] < 45:
        age_bin[i,1] = 1
    elif X[i,-2] >= 45 and X[i,-2] < 65:
        age_bin[i,2] = 1        

X_bin = np.concatenate([X_bin, age_bin, X[:,[4,-1]]], axis = 1)

labels = ["sbp 1", "sbp 2",
          "tobacco 1", "tobacco 2",
          "ldl 1", "ldl 2",
          "adiposity 1", "adiposity 2",
          "obesity 1", "obesity 2",
          "alcohol 1", "alcohol 2",          
          "age 1", "age 2", "age 3",
          "famhist",
          "chd"]

def mat2transactions(X, labels=[]):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X[i, :])[0].tolist()
        if labels:
            l = [labels[i] for i in l]
        T.append(l)
    return T

def print_apriori_rules(rules):
    frules = []
    for r in rules:
        for o in r.ordered_statistics:        
            conf = o.confidence
            supp = r.support
            x = ", ".join( list( o.items_base ) )
            y = ", ".join( list( o.items_add ) )
            print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)"%(x,y, supp, conf))
            frules.append( (x,y) )
    return frules

T = mat2transactions(X_bin,labels)
rules = apriori(T, min_support=0.8, min_confidence=0.95)
print_apriori_rules(rules)
print(medvalues)