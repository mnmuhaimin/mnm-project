#pandas read.csv() can parse string into datetime format
#needs to use parse_dates=True
#if datetime is used as index, we can select any data that contain certain datetime
sales.loc['February 5, 2015']
sales.loc['2015-Feb-5'] #same as above
sales.loc['2015-2'] #whole month
sales.loc['2015'] #whole year
sales.loc['2015-2-16':'2015-2-20'] #limit the time

#example 1
# Prepare a format string: time_format
time_format = '%Y-%m-%d %H:%M'

# Convert date_list into a datetime object: my_datetimes
#date_list format = 20100211 15:00
my_datetimes = pd.to_datetime(date_list, format=time_format)  

# Construct a pandas Series using temperature_list and my_datetimes: time_series
time_series = pd.Series(temperature_list, index=my_datetimes)

#example 2
#extract the data based on parameter
# Extract the hour from 9pm to 10pm on '2010-10-11': ts1
ts1 = ts0.loc['2010-10-11 21:00:00']

# Extract '2010-07-04' from ts0: ts2
ts2 = ts0.loc['2010-07-04']

# Extract data from '2010-12-15' to '2010-12-31': ts3
ts3 = ts0.loc['2010-12-15':'2010-12-31']


#example 3
#ts1 contains weekends but ts2 doesn't, so we need to reindex accordingly to the first, and combine them
#fill any NA by using ffill, so it will following the value of the nearest index, before the NA.
# Reindex without fill method: ts3
ts3 = ts2.reindex(ts1.index)

# Reindex with fill method, using forward fill: ts4
ts4 = ts2.reindex(ts1.index, method='ffill')

# Combine ts1 + ts2: sum12
sum12 = ts1 + ts2

# Combine ts1 + ts3: sum13
sum13 = ts1 + ts3

# Combine ts1 + ts4: sum14
sum14 = ts1 + ts4

#section 2
#downsampling: reduce datetime rows to slower freq, example daily to weekly
#upsampling: increase datetime rows to faster freq, example daily to hourly
#example
daily_mean = sales.resample('D').mean()
#will show the data on daily basis, 'D' means day, then averaging the value
#resampling period parameter:
Input	Desc
min, T	minute
H	hour
D	day
B	business day
W	week
M	month
Q	quarter
A	annual/year
#we can also put int before the paramter, example for biweekly:
sales.loc[:, 'Units'].resample('2W').sum() #will return the sum of each biweekly data

#upsampling example
two_days = sales.loc['2015-2-4':'2015-2-5', 'Units'] #example of data in those 2 days in hourly format
two_days.resample('4H').ffill() #will change the data into every-4-hour format and fill the NA data with forward fill

#example 1, change daily data to 6 hour data
# Downsample to 6 hour data and aggregate by mean: df1
df1 = df['Temperature'].resample('6h').mean()

# Downsample to daily data and count the number of data points: df2
df2 = df['Temperature'].resample('D').count()

print(df1, df2)

#example 2
# Extract temperature data for August: august
august = df['Temperature']['2010-08']

# Downsample to obtain only the daily highest temperatures in August: august_highs
august_highs = august.resample('D').max()

# Extract temperature data for February: february
february = df['Temperature']['2010-02']

# Downsample to obtain the daily lowest temperatures in February: february_lows
february_lows = february.resample('D').min()


#example 3, use rolling for the aggregation
# Extract data from 2010-Aug-01 to 2010-Aug-15: unsmoothed
unsmoothed = df['Temperature']['2010-08-01':'2010-08-15']

# Apply a rolling mean with a 24 hour window: smoothed
smoothed = unsmoothed.rolling(window=24).mean()

# Create a new DataFrame with columns smoothed and unsmoothed: august
august = pd.DataFrame({'smoothed':smoothed, 'unsmoothed':unsmoothed})

# Plot both smoothed and unsmoothed data using august.plot().
august.plot()
plt.show()


#example 4
# Extract the August 2010 data: august
august = df['Temperature']['2010-08']

# Resample to daily data, aggregating by max: daily_highs
daily_highs = august.resample('D').max()

# Use a rolling 7-day window with method chaining to smooth the daily high temperatures in August
#for rolling, we can't change the day or hour, because it will following the datetime format that previously adjusted
daily_highs_smoothed = daily_highs.rolling(window=7).mean()
print(daily_highs_smoothed)

#
