import matplotlib.pyplot as plt
import pandas as pd

#We shall use pandas to read the csv file and processing the data for the histogram

df = pd.read_csv("housing_training.csv", header=None)

#Since we know that housing_training.csv has no header, we can assign an arbitary header starting at index 0
# Index 0 corresponds to the letter A, thus we use the df.rename function to rename the coloumns in the csv file for our hw


df.rename(columns={0: 'A' }, inplace=True)

#Here we add additional arguments to our histogram function to make it more visually appealing and easier to read
df.hist(column= ['A'], bins=10, grid=False, color='green', zorder=2)
plt.title('Histogram of column A')
plt.show()
