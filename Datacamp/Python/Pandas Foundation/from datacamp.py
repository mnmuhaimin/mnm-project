PYTHON CHEAT SHEET

#RANDOM NUMBER GENERATOR / RANDOM WALK

# Import numpy and set seed
import numpy as np
np.random.seed(123)

# Initialize random_walk
random_walk = [0]

# Complete the ___
for x in range(100) :
    # Set step: last element in random_walk
    step = random_walk[-1]


    # Roll the dice
    dice = np.random.randint(1,7)

    # Determine next step
    if dice <= 2:
        step = step - 1
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    # append next_step to random_walk
    random_walk.append(step)

# Print random_walk
print(random_walk)

======================================================================================

# Initialization
import numpy as np
np.random.seed(123)
random_walk = [0]

for x in range(100) : #max index in 100 steps
    step = random_walk[-1]
    dice = np.random.randint(1,7)

    if dice <= 2:
        step = max(0, step - 1) #max() to ensure that the step won't reach below 0 >> max(limit, action)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    random_walk.append(step)

print(random_walk)
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Plot random_walk
plt.plot(random_walk)

# Show the plot
plt.show()

=============================================================== 

#tuples:
#Like a list, but immutable.
#example: 
tuples_a = (2, 3, 4)

=============================================================== 

#docstring : describe what a function does/documentation
#example:
def square(value):
	"""return the square of a value""" #this is the docstring
	new_value = value ** 2
	return new_value
	
	
	
===============================================================	
#will return a value, not print the variable	
# Define shout with the parameter, word
def shout(word):
    """Return a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = word + '!!!'    

    # Replace print with return
    return shout_word

# Pass 'congratulations' to shout: yell
yell = shout('congratulations')

# Print yell
print(yell)

===============================================================	
#add entry to dictionary, if not exist create new entry, if exist +1
# Import pandas
import pandas as pd

# Import Twitter data as DataFrame: df
df = pd.read_csv('tweets.csv')

# Initialize an empty dictionary: langs_count
langs_count = {}

# Extract column from DataFrame: col
col = df['lang']

# Iterate over lang column in DataFrame
for entry in col:

    # If the language is in langs_count, add 1
    if entry in langs_count.keys():
        langs_count[entry] += 1
    # Else add the language to langs_count, set the value to 1
    else:
        langs_count[entry] = 1

# Print the populated dictionary
print(langs_count)

===============================================================	
#create dictionary from column

# Define count_entries()
def count_entries(df, *args):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    #Initialize an empty dictionary: cols_count
    cols_count = {}
    
    # Iterate over column names in args
    for col_name in args:
    
        # Extract column from DataFrame: col
        col = df[col_name]
    
        # Iterate over the column in DataFrame
        for entry in col:
    
            # If entry is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1
    
            # Else add the entry to cols_count, set the value to 1
            else:
                cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'lang', 'source')

# Print result1 and result2
print(result1)
print(result2)

=======================================================

#lambda function:
lambda x, y: x ** y
x, y = variable

#structure:
lambda var1, var2: function #very usable to apply function to sequence
#example:

nums = [48, 6, 9, 21, 1]
square_all = map(lambda num: num ** 2)
print(square_all) #will show the type of square all which is map object
print(list(square_all)) #result = [2304, 36, 81, 441, 1]

#another  example

# Define echo_word as a lambda function: echo_word
echo_word = (lambda word1, echo: word1 * echo)

# Call echo_word: result
result = echo_word('hey', 5)

=====================================================================

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Use filter() to apply a lambda function over fellowship: result
#check whether name of member contain more that 6 letters, then filter will be applied based on the lambda function
result = filter(lambda member: len(member) > 6 , fellowship)

# Convert result to a list: result_list
result_list = list(result)

# Convert result into a list and print it
print(result_list)

=====================================================================

# Import reduce from functools
from functools import reduce

# Create a list of strings: stark
stark = ['robb', 'sansa', 'arya', 'eddard', 'jon']

# Use reduce() to apply a lambda function over stark: result
result = reduce(lambda item1, item2: item1 + item2, stark)

# Print the result
print(result)

=====================================================================

#error handling in python use try-except

def sqrt(x):
    try:
        return x ** 0.5 #default function
    except:
        print('x must be an int or float') #error message anytime an error occured

#example

# Define shout_echo
def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""

    # Initialize empty strings: echo_word, shout_words
    echo_word = ''
    shout_words = ''

    # Add exception handling with try-except
    try:
        # Concatenate echo copies of word1 using *: echo_word
        echo_word = word1 * echo

        # Concatenate '!!!' to echo_word: shout_words
        shout_words = echo_word + '!!!'
    except:
        # Print error message
        print("word1 must be a string and echo must be an integer.")

    # Return shout_words
    return shout_words

# Call shout_echo
shout_echo("particle", echo="accelerator")


=====================================================================

#error handling exaple with raise>ValueError
# Define shout_echo
def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""

    # Raise an error with raise
    if echo < 0:
        raise ValueError('echo must be greater than 0')

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Concatenate '!!!' to echo_word: shout_word
    shout_word = echo_word + '!!!'

    # Return shout_word
    return shout_word

# Call shout_echo
shout_echo("particle", echo=5)

=====================================================================
#lambda function to slice a list, the check if any of the slicing contain certain tex

# Select retweets from the Twitter DataFrame: result
result = filter(lambda x: x[0:2] == 'RT', tweets_df['text'])

# Create list from filter object result: res_list
res_list = list(result)

# Print all retweets in res_list
for tweet in res_list:
    print(tweet)


=====================================================================

#combine error handling with raise to function

# Define count_entries()
def count_entries(df, col_name='lang'):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    # Raise a ValueError if col_name is NOT in DataFrame
    if col_name not in df.columns:
        raise ValueError('The DataFrame does not have a ' + col_name + ' column.')

    # Initialize an empty dictionary: cols_count
    cols_count = {}
    
    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over the column in DataFrame
    for entry in col:

        # If entry is in cols_count, add 1
        if entry in cols_count.keys():
            cols_count[entry] += 1
            # Else add the entry to cols_count, set the value to 1
        else:
            cols_count[entry] = 1
        
        # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df, col_name = 'lang')

# Print result1
print(result1)

==============================================================
#iterator vs iterable
#iter can be used to create iterator from iterable object

# Create an iterator for range(3): small_value
small_value = iter(range(3))

# Print the values in small_value
print(next(small_value))
print(next(small_value))
print(next(small_value))

# Loop over range(3) and print the values
for i in range(3):
    print(i)


# Create an iterator for range(10 ** 100): googol
googol = iter(range(10 ** 100))

# Print the first 5 values from googol
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))


#enumerate = function to create index for list, enum()
#zip= combine iterable object

#enumerate example
avengers = ['hawkeye', 'iron man', 'thor']
e = enumerate(avengers)
print(type(e)) #print the type of e
<class 'enumerate'> #output
e_list = list(e) #to extract the enumerate object, convert enumerate to list
print(e_list)
[(0, 'hawkeye'). (1, 'iron man'), (2, 'thor')]

#because enumerate change the object to iterable object, we need to loop through the data to extract it
for index, value in enumerate(avengers):
    print index, value

#output
0 hawkeye #auto index start from 0
1 iron man
2 thor

#change the index start number by using 'start= x' parameter inside the enumerate
for index, value in enumerate(avengers, start = 5):
    print index, value

#output
5 hawkeye #auto index start from 0
6 iron man
7 thor

=========================================================================

#zip example

avengers = ['hawkeye', 'iron man', 'thor']
names = ['barton', 'stark', 'odinson']

z = zip(avengers, names)
print(type(zip)) #print the type of zip
<class 'zip'> #output
zip_list = list(zip) #to extract the enumerate object, convert enumerate to list
print(zip_list)
[('hawkeye', 'barton'). ('iron man', 'stark'), ('thor', 'odinson')]

for z1, z2 in zip(avengers, names):
    print(z1, z2)

#output
hawkeye barton
iron man stark
thor odinson

z = zip(avengers, names)
print(*z)
#output
('hawkeye', 'barton') ('iron man', 'stark'), ('thor', 'odinson')

#create dictionary of zipped avengers list
avengers_dict = dict(z)

#unpack a tuple
z_unpack = zip(*z1)


#chunk = temporary repository to hold some data, usually for big data.

# Initialize an empty dictionary: counts_dict
counts_dict = {}

# Iterate over the file chunk by chunk
for chunk in pd.read_csv('tweets.csv', chunksize = 10): #chunksize command associated with pandas

    # Iterate over the column in DataFrame
    for entry in chunk['lang']:
        if entry in counts_dict.keys():
            counts_dict[entry] += 1
        else:
            counts_dict[entry] = 1

# Print the populated dictionary
print(counts_dict)


============================================================
#list comprehension > modify list, especially useful for append new component
#can only be done to iterable object
#list comprehension structures: 
[[output expression] for iterator variable in iterable]

nums = [12, 8, 21, 3, 16]
new_nums = [num + 1 for num in nums]

#same as this for loop

for num in nums:
    new_nums.append(num + 1)



#example
for num1 in range(0, 2):
    for num2 in range(6, 8):
        pairs_1.append(num1, num2)

print(pairs_1)

#applied to:
pairs_2 = [(num1, num2) for num1 in range(0, 2) for num2 in range(6, 8)]

#another example
doctor = ['house', 'cuddy', 'chase', 'thirteen', 'wilson']
#normal loop to get the first character
for doc in doctor:
    print(doc[0])
#list comprehension
[doc[0] for doc in doctor]

============================================================

#create matrix 5x5 

# Create a 5 x 5 matrix using a list of lists: matrix
matrix = [[col for col in range(5)] for row in range(5)]

# Print the matrix
for row in matrix:
    print(row)

============================================================

#condition in iterable
[ output expression for iterator variable in iterable if predicate expression ]

[num ** 2 for num in range if num % 2 == 0]
[0, 4, 16, 36, 64] #output

#condition in output expression
[num ** 2 if num % 2 == 0 else 0 for num in range(10)] #if the input is not even number, it will be replaced as 0
[0, 0, 4, 0, 16, 0, 36, 0, 64, 0]


============================================================

#dictionary comprehension
pos_neg = {num: -num for num in range(5)}
print(pos_neg)
{0: 0, 1: -1, 2: -2, 3: -3, 4: -4}

============================================================

#generator = assigned list comprehension. Basically list comprehension will create a list and can be assigned to a variable which will make that variable as a list.
#list comprehension will store the data, so it will take a big portion of your memory, so the solution is to create generator, which will store list to comprehension to a variable.
[num for num in range(10)] #>> list comprehension
(num for num in range(10)) #>> generator

#generator creation example
# Create generator object: result
result = (num for num in range(31))

# Print the first 5 values
print(next(result))
print(next(result))
print(next(result))
print(next(result))
print(next(result))

# Print the rest of the values
for value in result:
    print(value)

#EXAMPLE OF GENERATOR FUNCTION
#DIFFERENCE WITH GENERAL FUNCTION IS THE OUTPUT WILL BE PRODUCED BY USING YIELD

# Create a list of strings
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Define generator function get_lengths
def get_lengths(input_list):
    """Generator function that yields the
    length of the strings in input_list."""

    # Yield the length of a string
    for person in input_list:
        yield len(person)

# Print the values generated by get_lengths()
for value in get_lengths(lannister):
    print(value)




# Extract the created_at column from df: tweet_time
tweet_time = df['created_at']

# Extract the clock time: tweet_clock_time
tweet_clock_time = [entry[11:19] for entry in tweet_time if entry[17:19] == '19']

# Print the extracted times
print(tweet_clock_time)


========================================================================================================================

#generator implementation

#In the function read_large_file(), read a line from file_object by using the method readline(). Assign the result to data
#In the function read_large_file(), yield the line read from the file data.
#In the context manager, create a generator object gen_file by calling your generator function read_large_file() and passing file to it.
#Print the first three lines produced by the generator object gen_file using next().

# Define read_large_file()
def read_large_file(file_object):
    """A generator function to read a large file lazily."""

    # Loop indefinitely until the end of the file
    while True:

        # Read a line from the file: data
        data = file_object.readline()

        # Break if this is the end of the file
        if not data:
            break

        # Yield the line of data
        yield(data)
        
# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Create a generator object for the file: gen_file
    gen_file = read_large_file(file)

    # Print the first three lines of the file
    print(next(gen_file))
    print(next(gen_file))
    print(next(gen_file))

=======================================================================================

#split df by df, adding condition to feature, combine features to zip

# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)

# Get the first DataFrame chunk: df_urb_pop
df_urb_pop = next(urb_pop_reader)

# Check out the head of the DataFrame
print(df_urb_pop.head())

# Check out specific country: df_pop_ceb
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

# Zip DataFrame columns of interest: pops
pops = zip(df_pop_ceb['Total Population'], df_pop_ceb['Urban population (% of total)'])

# Turn zip object into list: pops_list
pops_list = list(pops)

# Print pops_list
print(pops_list)

=======================================================================================

#Use pd.read_csv() to read in the file 'ind_pop_data.csv' in chunks of size 1000. Assign the result to urb_pop_reader.
#Write a list comprehension to generate a list of values from pops_list for the new column 'Total Urban Population'. Use tup as the iterator variable. The output expression should be the product of the first and second element in each tuple in pops_list. Because the 2nd element is a percentage, you also need to either multiply the result by 0.01 or divide it by 100. In addition, note that the column 'Total Urban Population' should only be able to take on integer values. To ensure this, make sure you cast the output expression to an integer with int().
#Create a scatter plot where the x-axis are values from the 'Year' column and the y-axis are values from the 'Total Urban Population' column.

# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)

# Get the first DataFrame chunk: df_urb_pop
df_urb_pop = next(urb_pop_reader)

# Check out specific country: df_pop_ceb
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

# Zip DataFrame columns of interest: pops
pops = zip(df_pop_ceb['Total Population'], 
            df_pop_ceb['Urban population (% of total)'])

# Turn zip object into list: pops_list
pops_list = list(pops)

# Use list comprehension to create new DataFrame column 'Total Urban Population'
df_pop_ceb['Total Urban Population'] = [int((tup[0] * tup[1])/100) for tup in pops_list]

# Plot urban population data
df_pop_ceb.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()

=======================================================================================

#numpy for load data, and format the type of the dataset

# Import package
import numpy as np

# Assign filename to variable: file
file = 'digits.csv'

# Load file as array: digits
# load the data using loadtxt
digits = np.loadtxt(file, delimiter=',')

# Print datatype of digits
print(type(digits))

#=======================================================================================
#another example

# Import numpy
import numpy as np

# Assign the filename: file
file = 'digits_header.txt'

# Load the data: data
# \t = tab delimiter
#skiprows will skip the row based on the number that being inputted, in this case 1
#usecols will show the column based on the number being inputted
data = np.loadtxt(file, delimiter='\t', skiprows=1, usecols=[0, 2])

# Print data
print(data)

#=======================================================================================
#another example

# Assign filename: file
file = 'seaslug.txt'

# Import file: data
# dtype = change the type of data
# loadtext can only be used to single type data, so the entire dataset type has to be converted
data = np.loadtxt(file, delimiter='\t', dtype=str)

# Print the first element of data
print(data[0])

# Import data as floats and skip the first row: data_float
data_float = np.loadtxt(file, delimiter='\t', dtype=float, skiprows=1)

# Print the 10th element of data_float
print(data_float[9])

# Plot a scatterplot of the data
plt.scatter(data_float[:, 0], data_float[:, 1])
plt.xlabel('time (min.)')
plt.ylabel('percentage of larvae')
plt.show()


#genfromtxt > just like loadtxt, but if dtype is inputted as None, it will automatically detect the type of each column
data = np.genfromtxt('titanic.csv', delimiter=',', names=True, dtype=None)

#recfromcsv > default of delimiter is comma (,), names is True, and dtype is none. So it is basically the same as genfromtxt but easier

#=======================================================================================

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Assign filename: file
file = 'titanic_corrupt.txt'

# Import file: data
data = pd.read_csv(file, sep='\t', comment='#', na_values='Nothing')

# Print the head of the DataFrame
print(data.head())

# Plot 'Age' variable in a histogram
pd.DataFrame.hist(data[['Age']]) #take the column for the histogram
plt.xlabel('Age (years)') 
plt.ylabel('count')
plt.show()

#=======================================================================================

#import excel file
import pandas as pd
file = 'testing.xlsx'
data = pd.ExcelFile(file)
print(data.sheet_names) #print the names of excel sheet

#set a sheet as a dataframe
df1 = data.parse('name of the sheet')

#set a sheet as a dataframe based on the index of the sheet
df2 = data.parse(0)


#=======================================================================================

#load pickle file
# Import pickle package
import pickle

# Open pickle file and load data: d
with open('data.pkl', 'rb') as file:    #rb means read only and binary, so it can be read by the computer
    d = pickle.load(file)

# Print d
print(d)

# Print datatype of d
print(type(d))

#=======================================================================================

#example of loading excel file into panda
# Import pandas
import pandas as pd

# Assign spreadsheet filename: file
file = 'battledeath.xlsx'

# Load spreadsheet: xl
xl = pd.ExcelFile(file)

# Print sheet names
print(xl.sheet_names)

#=======================================================================================

#load sheet to dataframe
# Load a sheet into a DataFrame by name: df1
df1 = xl.parse('2004')

# Print the head of the DataFrame df1
print(df1.head())

# Load a sheet into a DataFrame by index: df2
df2 = xl.parse(0)

# Print the head of the DataFrame df2
print(df2.head())

#=======================================================================================

# Parse the first sheet and rename the columns: df1
df1 = xl.parse(0, skiprows=[0], names=['Country', 'AAM due to War (2002)'])

# Print the head of the DataFrame df1
print(df1.head())

# Parse the first column of the second sheet and rename the column: df2
df2 = xl.parse(1, parse_cols=[0], skiprows=[0], names=['Country'])

# Print the head of the DataFrame df2
print(df2.head())


#output:
                   Country  AAM due to War (2002)
    0              Albania               0.128908
    1              Algeria              18.314120
    2              Andorra               0.000000
    3               Angola              18.964560
    4  Antigua and Barbuda               0.000000
                   Country
    0              Albania
    1              Algeria
    2              Andorra
    3               Angola
    4  Antigua and Barbuda


#=======================================================================================

#import SAS files
import pandas as pd
from sas7bdat import SAS7BDAT

with SAS7BDAT('name_of_the_file.sas7bdat') as file:
    df_sas = file.to_data_frame()   #load sas file to dataframe


#import stata file
import pandas as pd
data = pd.read_stata('name_of_the_file.dta')

#=======================================================================================

#import HDF5 file
import h5py
filename = 'name_of_the_file.hdf5'
data = h5py.File(filename, 'r')

#check the structure of HDF5 file
for key in data.keys():
    print(key)

#output
meta
quality
strain

#print the column in meta
for key in data['meta'].keys():
    print(key)

#Output
Description
DescriptionURL
Detector
Duration

print(data['meta']['Description'].value, data['meta']['Detector'])

b'Strain data time series from LIGO' b'H1'

#=======================================================================================

#import matlab file
import scipy.io 
filename = 'workspace.mat'
mat = scipy.io.loadmat(filename)
print(type(mat))

<class 'dict'> #output
#the common type of matlab file is dictionary
#keys in python = MATLAB variable names
#values in python = object assigned to the corresponding MATLAB variable

#=======================================================================================

#sqlalchemy > python module to connect to a database
from sqlalchemy import create_engine
engine = create_engine('type_of_database:///name_of_the_db')
engine = create_engine('sqlite:///Northwind.sqlite') #example


#assign and print the name of table in database
# Import necessary module
from sqlalchemy import create_engine

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Save the table names to a list: table_names
table_names = engine.table_names()

# Print the table names to the shell
print(table_names)



#run SQL Query in python
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite://Northwind.sqlite')
con = engine.connect() #connect to the engine
rs = con.execute("SELECT * FROM Orders") #execute query to the connection and put it to a variable.
df = pd.DataFrame(rs.fetchall()) #convert the query result to a dataframe variable
con.close() #close connection.

#the output column will not resulted as the name of the column/attribute, so it needs to be modify
#so additional action needs to be done for the dataframe

from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')
con = engine.connect() #connect to the engine
rs = con.execute("SELECT * FROM Orders") #execute query to the connection and put it to a variable.
df = pd.DataFrame(rs.fetchall()) #convert the query result to a dataframe variable
df.columns = rs.keys() #fetch the keys as column
con.close() #close connection.


#context manager to fetch selected data
#so the connection keeps open, and we dont have to worry about not closing it
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')
with engine.connect() as con: #use connect to keep the connection open
    rs = con.execute("SELECT OrderID, OrderDate, ShipName FROM Orders")
    df = pd.DataFrame(rs.fetchmany(size = 5))
    df.columns = rs.keys()


#example of using context manager
# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("select LastName, Title from Employee")
    df = pd.DataFrame(rs.fetchmany(3))
    df.columns = rs.keys()

# Print the length of the DataFrame df
print(len(df))

# Print the head of the DataFrame df
print(df.head())


# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("select * from Employee where EmployeeID >= 6")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()

# Print the head of the DataFrame df
print(df.head())


#using pandas feature to run query
df = pd.read_sql_query('name of the query', engine)

#example
# Import packages
from sqlalchemy import create_engine
import pandas as pd

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Execute query and store records in DataFrame: df
df = pd.read_sql_query('select * from Album', engine)

# Print head of DataFrame
print(df.head())

#=======================================================================================

#automate file download

from urllib.request import urlretrieve
url = 'name of the webpage.csv'
urlretrieve(url, 'name of the file.csv')

#example
# Import package
from urllib.request import urlretrieve

# Import pandas
import pandas as pd

# Assign url of file: url
url = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'

# Save file locally
urlretrieve(url, 'winequality-red.csv')

# Read file into a DataFrame and print its head
df = pd.read_csv('winequality-red.csv', sep=';') #sep is the separator of each attribute
print(df.head())



#example of load file without have to save it locally
# Import packages
import matplotlib.pyplot as plt
import pandas as pd

# Assign url of file: url
url = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'

# Read file into a DataFrame: df
df = pd.read_csv(url, sep=';') #put the url directly to the dataframe

# Print the head of the DataFrame
print(df.head())

# Plot first column of df
pd.DataFrame.hist(df.ix[:, 0:1])
plt.xlabel('fixed acidity (g(tartaric acid)/dm$^3$)')
plt.ylabel('count')
plt.show()


#load excel file, then retrieve certain sheet
# Import package
import pandas as pd

# Assign url of file: url
url = 'http://s3.amazonaws.com/assets.datacamp.com/course/importing_data_into_r/latitude.xls'

# Read in all sheets of Excel file: xl
xl = pd.read_excel(url, sheetname = None)

# Print the sheetnames to the shell
print(xl.keys()) #get the name of each sheet

# Print the head of the first sheet (using its name, NOT its index)
print(xl['1700'].head()) #get the data from sheet 1700



#=======================================================================================


urlretrieve() >> get request from http

#get request using urllib from url
from urllib.request import urlopen, Request
url = "https://wikipedia.org/"
request = Request(url)
response = urlopen(request)
html = response.read()
response.close()

#get request using request
import request
url = "https:/wikipedia.org/"
r = request.get(url)
text = r.text

#example of get url using Request
# Import packages
from urllib.request import urlopen, Request

# Specify the url
url = "http://www.datacamp.com/teach/documentation"

# This packages the request
request = Request(url)

# Sends the request and catches the response: response
response = urlopen(request)

# Extract the response: html
html = response.read()

# Print the html
print(html)

# Be polite and close the response!
response.close()


#get request using requests, converting the format to html
# Import package
import requests

# Specify the url: url
url = "http://www.datacamp.com/teach/documentation"

# Packages the request, send the request and catch the response: r
r = requests.get(url)

# Extract the response: text
text = r.text

# Print the html
print(text)


#example of using BeautifulSoup
# Import packages
import requests
from bs4 import BeautifulSoup

# Specify url: url
url = "https://www.python.org/~guido/"

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Extracts the response as html: html_doc
html_doc = r.text

# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)

# Prettify the BeautifulSoup object: pretty_soup
pretty_soup = soup.prettify()

# Print the response
print(pretty_soup)




#example of get text from BeautifulSoup
# Import packages
import requests
from bs4 import BeautifulSoup

# Specify url: url
url = 'https://www.python.org/~guido/'

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Extract the response as html: html_doc
html_doc = r.text

# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)

# Get the title of Guido's webpage: guido_title
guido_title = soup.title

# Print the title of Guido's webpage to the shell
print(guido_title)

# Get Guido's text: guido_text
guido_text = soup.get_text()

# Print Guido's text to the shell
print(guido_text)


#scrape the link tag of a website
# Import packages
import requests
from bs4 import BeautifulSoup

# Specify url
url = 'https://www.python.org/~guido/'

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Extracts the response as html: html_doc
html_doc = r.text

# create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)

# Print the title of Guido's webpage
print(soup.title)

# Find all 'a' tags (which define hyperlinks): a_tags
a_tags = soup.find_all('a')

# Print the URLs to the shell
for link in a_tags:
    print(link.get('href'))

#=======================================================================================

JSON > always store it in dictionary
#load JSON, then print it out
import json
with open('snakes.json', 'r') as json_data:
    json_data = json.load(json_file)

type(json_data) >> dict #the type of the variable will be dictionary
for key, value in json_data.items():
    print(key + ':', value)

#load and print out json data
# Load JSON: json_data
with open("a_movie.json") as json_file:
    json_data = json.load(json_file)

# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])

#=======================================================================================

#connect to API
import requests
url = 'http://www.omdbapi.com/?t=hackers' #> t = hackers means pulling every movie titled 'hackers', parameters based on the API
r = requests.get(url)
json_data = r.json()
for key, value in json_data.items():
    print(key + ':', value)

#connect to API and pulling the data
# Import requests package
import requests

# Assign URL to variable: url
url = "http://www.omdbapi.com/?apikey=ff21610b&t=social+network"

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Print the text of the response
print(r.text)

#convert to json format
# Decode the JSON data into a dictionary: json_data
json_data = r.json()

# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])


#example to apply it to wiki API
# Import package
import requests

# Assign URL to variable: url
url = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza"

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Decode the JSON data into a dictionary: json_data
json_data = r.json()

# Print the Wikipedia page extract
pizza_extract = json_data['query']['pages']['24768']['extract']
print(pizza_extract)

#=======================================================================================

#access to twitter api using tweepy
import tweepy, json
access_token = "..."
access_token_secret = "..."
consumer_key = "..."
consumer_secret = "..."
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
#stream to filtered tweet
# Initialize Stream listener
l = MyStreamListener()
# Create you Stream object with authentication
stream = tweepy.Stream(auth, l)
# Filter Twitter Streams to capture data by the keywords:
stream.filter(track=['clinton', 'trump', 'sanders', 'cruz'])


#load twitter data to a dictionary
# Import package
import json

# String of path to file: tweets_data_path
tweets_data_path = "tweets.txt"

# Initialize empty list to store tweets: tweets_data
tweets_data = []

# Open connection to file
tweets_file = open(tweets_data_path, "r")

# Read in tweets and store in list: tweets_data
for line in tweets_file:
    tweet = json.loads(line)
    tweets_data.append(tweet)

# Close connection to file
tweets_file.close()

# Print the keys of the first tweet dict
print(tweets_data[0].keys())

#take the dictionary to pandas dataframe
# Import package
import pandas as pd

# Build DataFrame of tweet texts and languages
df = pd.DataFrame(tweets_data, columns=['text', 'lang'])

# Print head of DataFrame
print(df.head())


#text analysis, count occurences
import re

def word_in_text(word, tweet):
    word = word.lower()
    text = tweet.lower()
    match = re.search(word, tweet)

    if match:
        return True
    return False

# Initialize list to store tweet counts
[clinton, trump, sanders, cruz] = [0, 0, 0, 0]

# Iterate through df, counting the number of tweets in which
# each candidate is mentioned
for index, row in df.iterrows():
    clinton += word_in_text('clinton', row['text'])
    trump += word_in_text('trump', row['text'])
    sanders += word_in_text('sanders', row['text'])
    cruz += word_in_text('cruz', row['text'])

#plot the lists
# Import packages
import matplotlib.pyplot as plt
import seaborn as sns


# Set seaborn style
sns.set(color_codes=True)

# Create a list of labels:cd
cd = ['clinton', 'trump', 'sanders', 'cruz']

# Plot histogram
ax = sns.barplot(cd, [clinton, trump, sanders, cruz]) #cd will use the name of the list as the label
ax.set(ylabel="count")
plt.show()

#=======================================================================================
#cleaning data and exploratory data analysis
#common data problem
- inconsistent column names
- missing data
- outliers
- duplicate rows
- untidy
- need to process columns
- column types can signal unexpected data values

#useful pandas tools to explore the data
# Print the head of df
print(df.head())
# Print the tail of df
print(df.tail())
# Print the shape of df
print(df.shape)
# Print the columns of df
print(df.columns)
# Print the info of df
print(df.info())
df.describe() #only for numerical data

#count the frequency
#can be very useful for categorical data
df.country.value_counts(dropna=False) #country is the name of the colume
df['country'].value_counts(dropna=False) #another way

#bar plots > discrete data
#histograms > continous data
#ploting continous data
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Plot the histogram
df['Existing Zoning Sqft'].plot(kind='hist', rot=70, logx=True, logy=True)

# Display the histogram
plt.show()

#slice the data by using other column/feature, by using boxplot
# Import necessary modules
import pandas as pd
import matplotlib.pyplot as plt

# Create the boxplot
df.boxplot(column='initial_cost', by='Borough', rot=90)

# Display the plot
plt.show()


#creating scatter plot to compare two numerical feature
# Import necessary modules
import pandas as pd
import matplotlib.pyplot as plt

# Create and display the first scatter plot
df.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)
plt.show()

# Create and display the second scatter plot
df_subset.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)
plt.show()

#=======================================================================================

#principles of tidy data
- columns represent separate variables
- rows represent individual observations
- observationals units form tables

#melting the data to tidying it
#melting example
pd.melt(frame=df, id_vars='name', value_vars=['treatement a', 'treatment b'], var_name = 'treatment', value_name='result')

#another example
# Melt airquality: airquality_melt
airquality_melt = pd.melt(frame=airquality, id_vars=['Month', 'Day'])

# Print the head of airquality_melt
print(airquality_melt.head())


#another example
# Print the head of airquality
print(airquality.head())

# Melt airquality: airquality_melt
airquality_melt = pd.melt(frame=airquality, id_vars=['Month', 'Day'], value_vars=['Ozone', 'Solar.R', 'Wind', 'Temp'], var_name='measurement', value_name='reading')

# Print the head of airquality_melt
print(airquality_melt.head())

#result
#input
  Ozone  Solar.R  Wind  Temp  Month  Day
0   41.0    190.0   7.4    67      5    1
1   36.0    118.0   8.0    72      5    2
2   12.0    149.0  12.6    74      5    3
3   18.0    313.0  11.5    62      5    4
4    NaN      NaN  14.3    56      5    5

#output
   Month  Day measurement  reading
0      5    1       Ozone     41.0
1      5    2       Ozone     36.0
2      5    3       Ozone     12.0
3      5    4       Ozone     18.0
4      5    5       Ozone      NaN

melting: turn columns into rows
pivot: turn unique values into separate columns (a.k.a. combine unique rows into one column)

#example
#input

weather_tidy = weather.pivot_table(index = 'date', #will become row
                                   columns = 'element', 
                                   values = 'values', #will become the value of the pivot
                                   aggfunc = np.mean)

#example
# Print the head of airquality_melt
print(airquality_melt.head())

# Pivot airquality_melt: airquality_pivot
airquality_pivot = airquality_melt.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading')

# Print the head of airquality_pivot
print(airquality_pivot.head())

#input
    Month  Day measurement  reading
    0      5    1       Ozone     41.0
    1      5    2       Ozone     36.0
    2      5    3       Ozone     12.0
    3      5    4       Ozone     18.0
    4      5    5       Ozone      NaN

#output
measurement  Ozone  Solar.R  Temp  Wind
    Month Day                              
    5     1       41.0    190.0  67.0   7.4
          2       36.0    118.0  72.0   8.0
          3       12.0    149.0  74.0  12.6
          4       18.0    313.0  62.0  11.5
          5        NaN      NaN  56.0  14.3

#comparing two types of dataframe by pivoting it
# Pivot airquality_dup: airquality_pivot
airquality_pivot = airquality_dup.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading', aggfunc=np.mean)
#default aggregation of pivot is mean, so actually there's no need to include np.mean

# Reset the index of airquality_pivot
airquality_pivot = airquality_pivot.reset_index()

# Print the head of airquality_pivot
print(airquality_pivot.head())

# Print the head of airquality
print(airquality.head())


#melting
#dirty data example where the column contain multiple information
country year m014 m1524
0 AD 2000 0 0
1 AE 2000 2 4
2 AF 2000 52 228

#m014 is male 0-14
#m1524 is male 15-24
#meling the data so it will be tidier like this
pd.melt(frame=tb, id_vars=['country', 'year'])
#the column that is going to be fixed as python list needs to be inputted in id_vars
#while the others are left so it won't be melted
#output
country year variable value
0 AD 2000 m014 0
1 AE 2000 m014 2
2 AF 2000 m014 52
3 AD 2000 m1524 0
4 AE 2000 m1524 4
5 AF 2000 m1524 228

#next extract the sex from variable feature, then rename it as sex column
tb_melt['sex'] = tb_melt.variable.str[0] #treat it as a string then extract the first letter of the variable feature
#output
country year variable value sex
0 AD 2000 m014 0 m
1 AE 2000 m014 2 m
2 AF 2000 m014 52 m
3 AD 2000 m1524 0 m
4 AE 2000 m1524 4 m
5 AF 2000 m1524 228 m

#useful syntax to extract the column
df.columns

#another example for melting the same data
#create new column for gender and age
# Melt tb: tb_melt
tb_melt = pd.melt(tb, id_vars=['country', 'year'])

# Create the 'gender' column
tb_melt['gender'] = tb_melt.variable.str[0]

# Create the 'age_group' column
tb_melt['age_group'] = tb_melt.variable.str[1:]

# Print the head of tb_melt
print(tb_melt.head())

#output
country year variable value gender age_group
0 AD 2000 m014 0.0 m 014
1 AE 2000 m014 2.0 m 014
2 AF 2000 m014 52.0 m 014
3 AG 2000 m014 0.0 m 014
4 AL 2000 m014 2.0 m 014


#melting and splitting data string
# Melt ebola: ebola_melt
ebola_melt = pd.melt(ebola, id_vars=['Date', 'Day'], var_name='type_country', value_name='counts')

# Create the 'str_split' column
#type_country is the name of a column, rather than using ebole_melt['type_country'], the column is typed like a method
ebola_melt['str_split'] = ebola_melt.type_country.str.split('_')

# Create the 'type' column
#same as previous (type_country), str_split is a column that is treated like function
ebola_melt['type'] = ebola_melt.str_split.str.get(0)

# Create the 'country' column
ebola_melt['country'] = ebola_melt.str_split.str.get(1)

# Print the head of ebola_melt
print(ebola_melt.head())

#column name before:
Index(['Date', 'Day', 'Cases_Guinea', 'Cases_Liberia', 'Cases_SierraLeone',
'Cases_Nigeria', 'Cases_Senegal', 'Cases_UnitedStates', 'Cases_Spain',
'Cases_Mali', 'Deaths_Guinea', 'Deaths_Liberia', 'Deaths_SierraLeone',
'Deaths_Nigeria', 'Deaths_Senegal', 'Deaths_UnitedStates',
'Deaths_Spain', 'Deaths_Mali'],
dtype='object')

#output
Date Day type_country counts str_split type country
0 1/5/2015 289 Cases_Guinea 2776.0 [Cases, Guinea] Cases Guinea
1 1/4/2015 288 Cases_Guinea 2775.0 [Cases, Guinea] Cases Guinea
2 1/3/2015 287 Cases_Guinea 2769.0 [Cases, Guinea] Cases Guinea
3 1/2/2015 286 Cases_Guinea NaN [Cases, Guinea] Cases Guinea
4 12/31/2014 284 Cases_Guinea 2730.0 [Cases, Guinea] Cases Guinea


#combining data by using concatenate
concatenated = pd.concat([df1, df2])
#the output will have duplicate index because of its origin index
#output example
date variable value
0 2010-01-30 tmax 27.8
1 2010-01-30 tmin 14.3
0 2010-02-02 tmax 27.3
1 2010-02-02 tmin 14.4
#will create duplicate index when slicing the data, like:
concatenated = concatenated.loc[0, :]
#output
date variable value
0 2010-01-30 tmax 27.8
0 2010-02-02 tmax 27.3
#use ignore_index when concatenate(the default ignore is False)
concatenated = pd.concat([df1, df2], ignore_index=True)
#output
date variable value
0 2010-01-30 tmax 27.8
1 2010-01-30 tmin 14.3
2 2010-02-02 tmax 27.3
3 2010-02-02 tmin 14.4

#check the shape of data, the output will be: (row, column)
df.shape

#normal concat will combine base one the row, if you want to concat the column, apply the axis=1 in the pd.concat function (default axis is axis=0)
# Concatenate ebola_melt and status_country column-wise: ebola_tidy
ebola_tidy = pd.concat([ebola_melt, status_country], axis=1)

# Print the shape of ebola_tidy
print(ebola_tidy.shape)

# Print the head of ebola_tidy
print(ebola_tidy.head())

#ebola_melt
Date Day status_country counts
0 1/5/2015 289 Cases_Guinea 2776.0
1 1/4/2015 288 Cases_Guinea 2775.0
2 1/3/2015 287 Cases_Guinea 2769.0
3 1/2/2015 286 Cases_Guinea NaN
4 12/31/2014 284 Cases_Guinea 2730.0

#status country
status country
0 Cases Guinea
1 Cases Guinea
2 Cases Guinea
3 Cases Guinea
4 Cases Guinea

#output of combining the columns
Date Day status_country counts status country
0 1/5/2015 289 Cases_Guinea 2776.0 Cases Guinea
1 1/4/2015 288 Cases_Guinea 2775.0 Cases Guinea
2 1/3/2015 287 Cases_Guinea 2769.0 Cases Guinea
3 1/2/2015 286 Cases_Guinea NaN Cases Guinea
4 12/31/2014 284 Cases_Guinea 2730.0 Cases Guinea

#concat many files, by using glob function to find pattern based on file names(e.g. date on the file names)
import glob
csv_files = glob.glob('*.csv') #find all csv files regardless the name
print(csv_files)
['file5.csv', 'file2.csv'] #output
#create a looping process to store the dataframes into a single file
list_data = []
for filename in csv_files:
data = pd.read_csv(filename)
list_data.append(data) #append all the associated dataframes to the list

pd.concat(list_data) #concat all the dataframes to a single file

# Import necessary modules
import glob
import pandas as pd

# Write the pattern: pattern
pattern = '*.csv'

# Save all file matches: csv_files
csv_files = glob.glob(pattern)

# Print the file names
print(csv_files)

# Load the second file into a DataFrame: csv2
csv2 = pd.read_csv(csv_files[1])

# Print the head of csv2
print(csv2.head())

#combine the csv data
# Create an empty list: frames
frames = []

# Iterate over csv_files
for csv in csv_files:

# Read csv into a DataFrame: df
df = pd.read_csv(csv)

# Append df to frames
frames.append(df)

# Concatenate frames into a single DataFrame: uber
uber = pd.concat(frames)

# Print the shape of uber
print(uber.shape)

# Print the head of uber
print(uber.head())

#combination of the dataframes that has different order can't be done by using concatenate
#alternative is using merging (like join in sql)
#example of 2 dataframes
df 1
state population_2016
0 California 39250017
1 Texas 27862596
2 Florida 20612439
3 New York 19745289

df2
name ANSI
0 Florida FL
1 Texas TX
2 California CA
3 New York NY

#use the merge function the join the dataframes
#use state in df1, and name in df2 as the key
pd.merge(left=df1, right=df2, on=None, left_on='state', right_on='name') #on is none because there is no similar column, while left comes from the left dataframe key (df1) and right from right dataframe key)

state population_2016 name ANSI
0 California 39250017 California CA
1 Texas 27862596 Texas TX
2 Florida 20612439 Florida FL
3 New York 19745289 New York NY

#like in sql, dataframes can have relationship as: one-to-one, one-to-many, many-to-one

#datatypes
#how to know the types of each column:
print(df.dtypes)
#example of converting data types
df['treatment b'] = df['treatment b'].astype(str) #convert to string
df['sex'] = df['sex'].astype('category') #convert to categorical

#convert and cleaning missing value
df['treatment a'] = pd.to_numeric(df['treatment a'], errors='coerce') #will treat missing value as empty / NaN


#example of converting to categorical
# Convert the sex column to type 'category'
tips.sex = tips.sex.astype('category') #alternatively can be written as: tips.sex = tips['sex'].astype('category')

# Convert the smoker column to type 'category'
tips.smoker = tips.smoker.astype('category') #alternatively can be written as: tips.smoker = tips['smoker'].astype('category')

# Print the info of tips
print(tips.info())

#If you expect the data type of a column to be numeric (int or float), but instead it is of type object, this typically means that there is a non numeric value in the column, which also signifies bad data.

#convert to numerical, then convert any error/different value as NaN
# Convert 'total_bill' to a numeric dtype
tips['total_bill'] = pd.to_numeric(tips['total_bill'], errors='coerce')

# Convert 'tip' to a numeric dtype
tips['tip'] = pd.to_numeric(tips['tip'], errors='coerce')

# Print the info of tips
print(tips.info())


#using regex to extract and clean strings
import re
pattern = re.compile('\$\d*\.\d{2}') #state the regex
result = pattern.match('$17.89') #declare the string that will be matched
bool(result) #compare the result whether it is true or not
# Import the regular expression module
import re

# Compile the pattern: prog
prog = re.compile('\d{3}-\d{3}-\d{4}')

# See if the pattern matches
result = prog.match('123-456-7890')
print(bool(result))

# See if the pattern matches
result = prog.match('1123-456-7890')
print(bool(result))

#find all instances that contain the regex
#basic formula: re.findall('regex pattern', the string)
# Import the regular expression module
import re

# Find the numeric values: matches
matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')

# Print the matches
print(matches)

#some other pattern in regex
# Write the first pattern
pattern1 = bool(re.match(pattern='\d{3}-\d{3}-\d{4}', string='123-456-7890')) #integer within -
print(pattern1)

# Write the second pattern
pattern2 = bool(re.match(pattern='\$\d*\.\d{2}', string='$123.45')) #integer within $ and decimal(.)
print(pattern2)

# Write the third pattern
pattern3 = bool(re.match(pattern='\w*', string='Australia')) #string
print(pattern3)

#create function to change categorical data
# Define recode_sex()
def recode_sex(sex_value):

# Return 1 if sex_value is 'Male'
if sex_value == 'Male':
return 1

# Return 0 if sex_value is 'Female'
elif sex_value == 'Female':
return 0

# Return np.nan
else:
return np.nan

#use .apply to run the function to certain column, and create new column
# Apply the function to the sex column
tips['sex_recode'] = tips.sex.apply(recode_sex)

# Print the first five rows of tips
print(tips.head(5))

#use lambda function to apply replace function and findall regex
# Write the lambda function using replace
tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))

# Write the lambda function using regular expressions
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])

# Print the head of tips
print(tips.head())

#drop duplicates
df = df.drop_duplicates()

#example
# Create the new DataFrame: tracks
tracks = billboard[['year', 'artist', 'track', 'time']]

# Print info of tracks
print(tracks.info())

# Drop the duplicates: tracks_no_duplicates
tracks_no_duplicates = tracks.drop_duplicates()

# Print info of tracks
print(tracks_no_duplicates.info())


#drop missing value
df = df.dropna()

#filling missing value
tips['sex'] = tips['sex'].fillna('missing') #filling string type of data
tips[['total_bill', 'size']] = tips[['total_bill', 'size']].fillna(0) #filling integer type with 0, applied to multiple column, that's why we use double [] bracket
#filling with mean
mean_value = tips['tip'].mean()
tips['tip'] = tips['tip'].fillna(mean_value)
#example
# Calculate the mean of the Ozone column: oz_mean
oz_mean = airquality['Ozone'].mean()

# Replace all the missing values in the Ozone column with the mean
airquality['Ozone'] = airquality['Ozone'].fillna(oz_mean)

#validate the content of data (not null)
assert df.name_of_column.notnull().all() #the result will show an error if there is a null
# Assert that there are no missing values
assert pd.notnull(ebola).all().all()

# Assert that all values are >= 0
assert (ebola>=0).all().all()


#useful method
df = pd.read_csv('data.csv')
df.head()
df.info()
df.columns
df.describe()
df.column.value_counts() #check whether there's outlier or not in the data
df.column.plot('hist') #create histogram for each column
assert(df.column_data > 0).all() #validate the data
df.dtypes
df['column'].to_numeric()
df['column'].astype(str)


===================================IMPORTANT=============================
#important to cleaning and tidying data
#formatting data
#Define a function called check_null_or_valid() that takes in one argument row_data.
#Inside the function, convert no_na to a numeric data type using pd.to_numeric().
def check_null_or_valid(row_data):
"""Function that takes a row of data,
drops all missing values,
and checks if all remaining values are greater than or equal to 0
"""
no_na = row_data.dropna()[1:-1]
numeric = pd.to_numeric(no_na)
ge0 = numeric >= 0
return ge0

# Check whether the first column is 'Life expectancy'
#Write an assert statement to make sure the first column (index 0) of the g1800s DataFrame is 'Life expectancy'.
assert g1800s.columns[0] == 'Life expectancy'

# Check whether the values in the row are valid
#Write an assert statement to test that all the values are valid for the g1800s DataFrame. Use the check_null_or_valid() function placed inside the .apply() method for this. Note that because you're applying it over the entire DataFrame, and not just one column, you'll have to chain the .all() method twice, and remember that you don't have to use () for functions placed inside .apply().
assert g1800s.iloc[:, 1:].apply(check_null_or_valid, axis=1).all().all()

# Check that there is only one instance of each country
#Write an assert statement to make sure that each country occurs only once in the data. Use the .value_counts() method on the 'Life expectancy' column for this. Specifically, index 0 of .value_counts() will contain the most frequently occuring value. If this is equal to 1 for the 'Life expectancy' column, then you can be certain that no country appears more than once in the data.
assert g1800s['Life expectancy'].value_counts()[0] == 1

#tidy data principle
#rows form observations
#columns form variable
#melting turns columns into rows
#pivot will take unique value from a column and create new columns

#melt the data that has been combined
# Melt gapminder: gapminder_melt
gapminder_melt = pd.melt(gapminder, id_vars='Life expectancy')

# Rename the columns
gapminder_melt.columns = ['country', 'year', 'life_expectancy']

# Print the head of gapminder_melt
print(gapminder_melt.head())

# Convert the year column to numeric
gapminder.year = pd.to_numeric(gapminder['year'])

# Test if country is of type object
assert gapminder.country.dtypes == np.object

# Test if year is of type int64
assert gapminder.year.dtypes == np.int64

# Test if life_expectancy is of type float64
assert gapminder.life_expectancy.dtypes == np.float64

# Create the series of countries: countries
countries = gapminder['country']

# Drop all the duplicates from countries
countries = countries.drop_duplicates()

# Write the regular expression: pattern
pattern = '^[A-Za-z\.\s]*$'
#find country that does not contain this regex pattern:

# The set of lower and upper case letters.
# Whitespace between words.
# Periods for any abbreviations.


# Create the Boolean vector: mask
mask = countries.str.contains(pattern)

# Invert the mask: mask_inverse, ~ will invert the boolean
mask_inverse = ~mask

# Subset countries using mask_inverse: invalid_countries
invalid_countries = countries.loc[mask_inverse]

# Print invalid_countries
print(invalid_countries)

# Assert that country does not contain any missing values
assert pd.notnull(gapminder.country).all()

# Assert that year does not contain any missing values
assert pd.notnull(gapminder.year).all()

# Drop the missing values
gapminder = gapminder.dropna()

# Print the shape of gapminder
print(gapminder.shape)

# Add first subplot
plt.subplot(2, 1, 1)

# Create a histogram of life_expectancy
gapminder.life_expectancy.plot(kind='hist')

# Group gapminder: gapminder_agg
gapminder_agg = gapminder.groupby('year')['life_expectancy'].mean()

# Print the head of gapminder_agg
print(gapminder_agg.head())

# Print the tail of gapminder_agg
print(gapminder_agg.tail())

# Add second subplot
plt.subplot(2, 1, 2)

# Create a line plot of life expectancy per year
gapminder_agg.plot()

# Add title and specify axis labels
plt.title('Life expectancy over the years')
plt.ylabel('Life expectancy')
plt.xlabel('Year')

# Display the plots
plt.tight_layout()
plt.show()

# Save both DataFrames to csv files
gapminder.to_csv('gapminder.csv')
gapminder_agg.to_csv('gapminder_agg.csv')