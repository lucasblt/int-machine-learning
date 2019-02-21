# exercise 1.5.1
import numpy as np
import pandas as pd

# Load the Iris csv data using the Pandas library
filename = 'https://web.stanford.edu/~hastie/ElemStatLearn//datasets/SAheart.data'
df = pd.read_csv(filename)

raw_data = df.get_values() 

cols = range(0, 4) 
X = raw_data[:, cols]

attributeNames = np.asarray(df.columns[cols])

classLabels = raw_data[:,-1]


famHistNames = raw_data[:,5]
famHistLabels = np.unique(famHistNames)

famHistDict = dict(zip(famHistLabels,[0,1]))

#List comprehension on attribute famhist
famHistLc = np.array([famHistDict[cl] for cl in famHistNames])


#Insert the list comprehension for column famhist in the raw_data
new_data = np.hstack((raw_data[:,0:5],np.array([famHistLc]).T,raw_data[:,6:]))

print(pd.DataFrame(new_data))
