# Sarkis Gafafyan



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# We can visualize the MNIST data using PCA, here I have setup proper x & y values to be used in the PCA function
data = pd.read_csv('MNIST_100.csv')
y = data.iloc[:,0]
x = data.drop('label', axis=1)
print(x.shape)
print(y.shape)

# We can now apply the PCA function to the data and plot the results
pca = PCA(n_components=2)
pca.fit(x) # Here we compute the SVD
PCAX = pca.transform(x) # Transform this into a 2D principal component space

plt.plot(PCAX[:, 0], PCAX[:, 1], 'wo', ) # We can start plotting the data and adding labels
for i in range(len(y)): # Lets iterate through y to add labels to the data
    plt.text(PCAX[i:i+1, 0], PCAX[i:i+1, 1], y[i])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of MNIST data')
plt.show()