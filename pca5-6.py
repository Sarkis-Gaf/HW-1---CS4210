import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# We can visualize the MNIST data using PCA, here I have setup proper x & y values to be used in the PCA function
data = pd.read_csv('MNIST_100.csv')
# To make just 5 & 6 show we must filter the data using the isin function

filteredData = data[data['label'].isin([5, 6])]

y = filteredData.iloc[:,0]
x = filteredData.drop('label', axis=1)
print(x.shape)
print(y.shape)

# We can now apply the PCA function to the data and plot the results
pca = PCA(n_components=2)
pca.fit(x) # Here we compute the SVD
PCAX = pca.transform(x) # Transform this into a 2D principal component space

plt.plot(PCAX[:, 0], PCAX[:, 1], 'wo', ) # We can start plotting the data and adding labels
for i in range(len(y)): # Lets iterate through y to add labels to the data
    plt.text(PCAX[i, 0], PCAX[i, 1], str(y.iloc[i])) # This fixes any indexing issues for 5-6
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of MNIST data of 5 & 6')
plt.show()