import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("housing_training.csv", header=None)

print(df.head())

df.rename(columns={10: 'K', 12: 'M', 13: 'N'}, inplace=True)

# Here I created a rename function to change the column names to match with the corresponding coloum names found after printing the data
# I then created a boxplot of the data using the boxplot function from pandas

df.boxplot(column= ['K', 'M', 'N'])

#We can use searborn to visualize our basic boxlpot

sns.boxplot(data=df[['K', 'M', 'N']])
sns.stripplot(data=df[['K', 'M', 'N']], jitter=0.1, size= 3,  color='grey')

#Utilizing searborn allows us to add more features to our boxplot, which helps us visualize the data better

plt.title('Boxplot of K, M, N')
plt.show()