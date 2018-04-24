#using iris dataset
import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv('iris.csv', index_col=0)
print(iris.shape)
#column name: sepal_length, sepal_width, petal_length, petal_width

iris.head()
iris.plot(x='sepal_length', y='sepal_width')
plt.show()
#it won't result it a good plot, so we will change plot to scatter
iris.plot(x='sepal_length', y='sepal_width',kind='scatter')
plt.xlabel('sepal length (cm)')
plt.xlabel('sepal width (cm)')
plt.show()
#next we try to box plot each of the column
iris.plot(y='sepal_length', kind='box')
plt.ylabel('sepal length (cm)')
plt.show()
#then createing histogram
iris.plot(y='sepal_length', kind='hist')
plt.ylabel('sepal length (cm)')
plt.show()
#customizing the histogram
iris.plot(y='sepal_length', kind='hist', bins=30, range=(4,8), normed=True)
plt.ylabel('sepal length (cm)')
plt.show()

#three kind of plotting
iris.plot(kind='hist')
iris.plt.hist()
iris.hist()
#can be used to all plot, but first check documentation to ensure it

#example 1
# Create a list of y-axis column names: y_columns
y_columns = ['AAPL', 'IBM']

# Generate a line plot
df.plot(x='Month', y=y_columns)

# Add the title
plt.title('Monthly stock prices')

# Add the y-axis label
plt.ylabel('Price ($US)')

# Display the plot
plt.show()

#output
	Month	AAPL	GOOG	IBM
0	Jan	117.160004	534.522445	153.309998
1	Feb	128.460007	558.402511	161.940002
2	Mar	124.43	548.002468	160.5
3	Apr	125.150002	537.340027	171.289993
4	May	130.279999	532.109985	169.649994

#example 2, create scatter plot
# Generate a scatter plot
df.plot(kind='scatter', x='hp', y='mpg', s=sizes) #s is the parameter to set the size of scatter plot

# Add the title
plt.title('Fuel efficiency vs Horse-power')

# Add the x-axis label
plt.xlabel('Horse-power')

# Add the y-axis label
plt.ylabel('Fuel efficiency (mpg)')

# Display the plot
plt.show()

#3rd example
#create box subplots for two columns that has different unit
# Make a list of the column names to be plotted: cols
cols = ['weight', 'mpg']

# Generate the box plots
df[cols].plot(kind='box', subplots=True)

# Display the plot
plt.show()

#4th example
#create PDF and CDF
#PDF means the normed has to be true, while for CDF normed has to be true and cumulative also true
#PDF: probability density function
#CDF: cumulative distribution function
# This formats the plots such that they appear on separate rows
fig, axes = plt.subplots(nrows=2, ncols=1)

# Plot the PDF
df.fraction.plot(ax=axes[0], kind='hist', normed=True, bins=30, range=(0,.3))
plt.show()

# Plot the CDF
df.fraction.plot(ax=axes[1], kind='hist', normed=True, bins=30, cumulative=True, range=(0,.3))
plt.show()


#new section: explore the statistics of the data
#4th example
# Print the minimum value of the Engineering column
print(df['Engineering'].min())

# Print the maximum value of the Engineering column
print(df['Engineering'].max())

# Construct the mean percentage per year: mean
#year is the index, so in order to calculate the mean, the axis is stated as 'columns'
mean = df.mean(axis='columns')

# Plot the average percentage per year
mean.plot()

# Display the plot
plt.show()

#2nd example
#boxplot to explore outliers, and why median will be more useful in the presence of outliers
# Print summary statistics of the fare column with .describe()
print(df['fare'].describe())

# Generate a box plot of the fare column
df['fare'].plot(kind='box')

# Show the plot
plt.show()

#3rd example
#check the 5% and 95% quantile then look up the trends
# Print the number of countries reported in 2015
print(df['2015'].count())

# Print the 5th and 95th percentiles
print(df.quantile([0.05, 0.95]))

# Generate a box plot
years = ['1800','1850','1900','1950','2000']
df[years].plot(kind='box')
plt.show()

#4th example
# Print the mean of the January and March data
print(january.mean(), march.mean())

# Print the standard deviation of the January and March data
print(january.std(), march.std())


#new section
#separate discrete value (could be string) with boolean
#column speciaes in iris is alphabetical so it has to be treated in other way
iris['species'].unique() #will produce array of each element in species column

#extract each element to create a new dataframe
indices = iris['species'] == 'setosa'
setosa = iris.loc[indices,:] #new dataframe
indices = iris['species'] == 'versicolor'
versicolor = iris.loc[indices,:]
indices = iris['species'] == 'virginica'
virginica = iris.loc[indices,:]

#1st example
# Compute the global mean and global standard deviation: global_mean, global_std
global_mean = df.mean()
global_std = df.std()

# Filter the US population from the origin column: us
us = df[df['origin'] == 'US']

# Compute the US mean and US standard deviation: us_mean, us_std
us_mean = us.mean()
us_std = us.std()

# Print the differences
print(us_mean - global_mean)
print(us_std - global_std)

#2nd example
# Display the box plots on 3 separate rows and 1 column
fig, axes = plt.subplots(ncols=1, nrows=3)

# Generate a box plot of the fare prices for the First passenger class
# y= will be the name of label
titanic.loc[titanic['pclass'] == 1].plot(ax=axes[0], y='fare', kind='box')

# Generate a box plot of the fare prices for the Second passenger class
titanic.loc[titanic['pclass'] == 2].plot(ax=axes[1], y='fare', kind='box')

# Generate a box plot of the fare prices for the Third passenger class
titanic.loc[titanic['pclass'] == 3].plot(ax=axes[2], y='fare', kind='box')

# Display the plot
plt.show()