# Zip the 2 lists together into one list of (key,value) tuples: zipped
# in python 3 zip always has to be listed, because zip return an iterator
zipped = list(zip(list_keys, list_values))

# Inspect the list using print()
print(zipped)

# Build a dictionary with the zipped list: data
data = dict(list(zipped))

# Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)
print(df)


#adding column name to headerless data
column_names = ['year', 'month', 'day', 'dec_date', 'sunspots', 'definite']
sunspots = pd.read_csv(filepath, header=None, names=column_names) #column names list will be used as the name of the header

#manipulating na values

sunspots = pd.read_csv(filepath, header=None, names=column_names, na_values = ' -1') #-1 will be treated as NaN, so if you have values that you want to treat as NA, use na_values
#another way by using column names as dictionary key, then treat the values as NaN
sunspots = pd.read_csv(filepath, header=None, names=column_names, na_values = {'sunspots':[' -1']}) 

#example of renaming column
# Read in the file: df1
df1 = pd.read_csv('world_population.csv')

# Create a list of the new column labels: new_labels
new_labels = ['year', 'population']

# Read in the file, specifying the header and names parameters: df2
df2 = pd.read_csv('world_population.csv', header=0, names=new_labels)

# Print both the DataFrames
print(df1)
print(df2)

#plotting with matplotlibs
import pandas as pd
import matplotlib.pyplot as plt

aapl = pd.read_csv(filename, index_col = True, parse_dates = True)
aapl.head(3)

#output
date    adj_close   close   high    low open    volume
01/03/2000  31.68   130.31  132.06  118.5   118.56  38478000
02/03/2000  29.66   122 127.94  120.69  127 11136800

close_arr = aapl['close'].values #get values of close feature
type(close_arr) #output = numpy.ndarray
plt.plot(close_arr) #plot the numpy array
plt.show(close_arr) #show the plot

#will show the data but the x axis is not showing the time correctly
#another way is to use series from pandas

close_series = aapl['close'] #no need to get the value
type(close_series) #output = pandas.core.series.Series
plt.plot(close_series) #plot it
plt.show(close_series) #show the plot

#faster way to plotting, will also show the name of the x axis
close_series.plot()
plt.show()

#pandas can be used to plot all the feature
aapl.plot()
plt.show() #the result might be weird because of the axis scale

#solution: create logarithmic scale
aapl.plot()
plt.yscale('log')
plt.show()

#saving plot
aapl.loc['2001':'2004,'['open','close','high','low']].plot()
plt.savefig(aapl.jpg)
plt.savefig(aapl.png)
plt.savefig(aapl.pdf)
plt.show()

#plotting training
# Create a plot with color='red'
df.plot(color='red')

# Add a title
plt.title('Temperature in Austin')

# Specify the x-axis label
plt.xlabel('Hours since midnight August 1, 2010')

# Specify the y-axis label
plt.ylabel('Temperature (degrees F)')

# Display the plot
plt.show()

#another plotting example
# Plot all columns (default)
df.plot()
plt.show()

# Plot all columns as subplots
#subplots will plot all the column into single chart
df.plot(subplots=True)
plt.show()

# Plot just the Dew Point data
column_list1 = ['Dew Point (deg F)']
df[column_list1].plot()
plt.show()

# Plot the Dew Point and Temperature data, but not the Pressure data
column_list2 = ['Temperature (deg F)','Dew Point (deg F)']
df[column_list2].plot()
plt.show()