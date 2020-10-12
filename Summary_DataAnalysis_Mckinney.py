#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Purpose of the project is to create a repo for using the basics of pandas dataframes 


# ## Dataframes

# In[2]:


import pandas as pd 
import numpy as np


# ### How can we create dataframes?

# In[3]:


data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
'year': [2000, 2001, 2002, 2001, 2002, 2003],
'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)


# In[4]:


frame


# In[5]:


pd.DataFrame(data, columns= ['year', 'state', 'pop']) #select the order of the columns appearing in the dataframe


# In[6]:


frame2 = pd.DataFrame(data, columns = ['year', 'state', 'pop'], index = [1,2,3,4,5,6]) # we can also assign the indices to suit our needs


# In[7]:


frame2


# In[8]:


frame2.year  # we can print the relevant elements within a dataframe 


# In[9]:


frame2.index


# In[10]:


frame2.columns


# In[11]:


frame2['year'] #dot or [] syntax can be used to print columns


# In[12]:


#for finding particular elements within a data frame there are 3 functions we can use: 
#iloc, loc, ix


# In[13]:


frame2.loc[1]


# In[14]:


frame2['debt'] = np.arange(6.)


# In[15]:


val2  = pd.Series([-1.2, -1.5, -1.7], index = [2,4,5]) #we can insert a pd series as a colomn in a dataframe


# In[16]:


frame2['debt'] = val2


# In[17]:


frame2 #this will give us nan values where we have incompletely assigned the indices


# In[18]:


frame2['eastern'] = frame2.state == 'Ohio'
frame2                                      #boolean values can also be inserted
                                            #New columns cannot be created with the frame2.eastern syntax.


# In[19]:


del frame2['eastern']


# In[20]:


pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}  #nested dictionaries can also be used, outer keys act as columns and inner keys as indices


# In[21]:


frame3 = pd.DataFrame(pop)
frame3


# In[22]:


pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}  #we can revert back by adding the data inside a new dictionary and performing indexing in the first dataframe


# In[23]:


pd.DataFrame(pdata)


# In[24]:


frame3.index.name = "year"
frame3.columns.name = "state"
frame3


# In[25]:


frame3.values


# In[26]:


#Index Objects: these are responsible for holding the axis labels and other metadata (e.g. axis names)
#Note that indices are imutable objects 


# In[27]:


labels = pd.Index(np.arange(3))


# In[28]:


obj2 = pd.Series([-1.2, -1.5, -1.7], index = labels)


# In[29]:


obj2.index is labels # indices can have duplicate values


# In[30]:


frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
        index=['a', 'c', 'd'],
        columns=['Ohio', 'Texas', 'California']) #we can also reindex a dataframe


# In[31]:


states = ['Utah', 'Ohio', 'Texas']


# In[32]:


#we can 'drop' data by using the drop() method: name_of_dataframe.drop([list of elements], axis = 0 |1, inplace = Trues|False )


# In[33]:


#The stansard way to find elements in a datframe is via loc(labels) and iloc(integers), let's have a look at some examples 


# In[34]:


data = pd.DataFrame(np.arange(16).reshape((4, 4)),
       index=['Ohio', 'Colorado', 'Utah', 'New York'],
       columns=['one', 'two', 'three', 'four'])


# In[35]:


data.loc['Ohio', ['one', 'two']]


# In[36]:


data.loc[['Ohio', 'Colorado'], ['one','two']]


# In[37]:


data.iloc[0, [0,1]]


# In[38]:


data.loc['Ohio':'Colorado', 'one':] #slicing works with both loc and iloc


# In[39]:


data.iloc[0:2, 0:] #not inclusive slicing in iloc


# In[40]:


data.iloc[0:, 0:] [data.two >0]


# In[41]:


# Indexing, slicing and filtering in Dataframes and Series


# In[42]:


data[['one', 'two', 'three']]


# In[43]:


#Indexing like the above comes with special cases


# In[44]:


data[0:2] # there is no indexing version for the rows  -> keyerror


# In[45]:


data[data['one'] >5]


# In[46]:


data < 5 #This will result in a boolean array 


# In[47]:


#for series what changes from the above notation is that slicing is inclusive 
#also remember that slicing is usually a  view while indexing creates a copy of the initial array-like object


# ### Integer Indexes 

# In[48]:


ser = pd.Series(np.arange(3.))
ser
ser[-1]  # -> error keytype


# In[49]:


#to be consistent in this case we have integer index values it preferable to use loc or iloc


# ### Apply and mapping 

# In[50]:


frame = pd.DataFrame(np.random.randn(4,3), index = list('abcd'), columns = ["Utah", "Ohio", "Texas"])
frame 


# In[51]:


f = lambda x: x.max() - x.min() 


# In[52]:


frame.apply(f) # applies f per column 


# In[53]:


frame.apply(f, axis = 1) #same as calling axis = "columns"


# In[54]:


#we can also return apart a scalar value a one dimensional array i.e. a Series


# In[55]:


def f(x):
    return pd.Series([x.max(), x.min(), x.sum()], index = ['max', 'min', 'sum'])


# In[56]:


frame.apply(f)


# In[57]:


#we can parse through each element performing a predefined operation by using the applymap() method


# In[58]:


round_to_two = lambda x: "%.2f" %x 


# In[59]:


frame.applymap(round_to_two)


# In[60]:


#for perfrorming the same operations on a Series we use the map() method


# In[61]:


frame["Utah"].map(round_to_two)


# ### Sorting and Ranking

# In[62]:


obj = pd.Series(range(4), index = [ 'b', 'a','c', 'd'])
obj 


# In[63]:


obj.sort_index() # we can also have the option to sort by values: sort_values()


# In[64]:


frame = pd.DataFrame(np.arange(8).reshape(2,4), index = ['a','b'], columns = ["Utah", "Ohio", "Texas" , "Oregon"] )
frame


# In[65]:


frame.sort_index(axis=1)


# In[66]:


frame.sort_index(axis = 1, ascending  = False)


# In[67]:


frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame


# In[68]:


frame.sort_values(by= ['b','a']) # we can sort a dataframe by the values of one or more than one of its columns


# ### Descriptive statistics 

# options for reductions methods (sum, mean, etc): 
#   - axis (0, 1)
#   - skipna (False, True)
#   - level 

# idmax(), idmin() can be used to find the maximum and minimum values in each column of the dataframe 

#  cumsum() can perform an accumulation over each of the columns 
# 

# describe() give us a general overview of our data within our dataframe (mean, deviation, percentiles, count)

# corr() and cov() can be used in Series as well as a Dataframe

# <h2> Grouping data </h2>

# <h3> Group by Mechanics </h3>

# In[69]:


import psycopg2
conn  = psycopg2.connect(dbname = 'postgres', user = 'postgres', password = 'durham17')
conn.close()


# In[70]:


import pandas as pd 
import numpy as np

df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
     'key2' : ['one', 'two', 'one', 'two', 'one'],
     'data1' : np.random.randn(5),
     'data2' : np.random.randn(5)})


# In[71]:


grouped = df['data1'].groupby(df['key1']) #identify the column on which we will perform the grouping 
#                                            this will usuallly be a colummn within the same dataframe
#                                            therefore, a simple 'key1' also makes up for it


# In[72]:


grouped #nothing yet calculated, the object is simply now a Grouped object
        #allowing us to perform any operation on the grouped data like sum, avg etc on the group 


# In[73]:


grouped.sum()


# In[74]:


new_grouped = df['data1'].groupby([df['key1'], df['key2']]).sum()
new_grouped #usually you want to perform the grouping in terms of a column 
            #alrady existing in the dataframe, hence you just pass the labels as an argument 
            #in the group by function


# In[75]:


new_grouped.unstack() #since we have multiple layers of indices we can play around by unstacking 


# In[76]:


df.groupby(['key1', 'key2']).size() #count the elements within each group


# In[77]:


for (k1,k2), group in df.groupby(['key1', 'key2']):
    print(k1,k2)
    print(group)
#groupby object is iterable, therefore we can parse it with a for 
#getting the each separate key and its chunk of data


# In[78]:


pieces = list(df.groupby('key1')) #since groupby is iterable we cast the object into list like 
                                  #objects i.e. lists, tuples, dicts 


# In[79]:


pieces


# In[80]:


pieces = dict(pieces)


# In[81]:


pieces


# In[82]:


grouped = df.groupby(df.dtypes, axis = 1) #remember that groupby groups by default on 0 axis 
                                          #that said, along the rows

for data_types, data in grouped: #group data by their data types
    print (data_types)
    print(data)


# In[83]:


for data_types, data in grouped: #group data by their data types
    print (data_types)
    print(data)


# In[84]:


df.groupby('key1').sum()


# In[85]:


df.groupby('key1')['data1'].sum() #brings out a series


# In[86]:


df.groupby('key1')[['data1']].sum() #brings out a dataframe


# <h3> Data Aggregation </h3>

# <p>  Data aggregation functions are simply the functions that act on multidimensional 
#     arrays to return a scalar value. Functions like these inclde sum(), average(), count(), 
#     min(), max(), median(), var(), std(), etc...
#    

# <p> One can define his own aggregation function by defining the function itself and then passing its process to the data via the .agg() method e.g.:
#     
#     def peak_to_peak(arr):
#         return arr.max() - arr.min()
#     grouped.agg(peak_to_peak) 
#     
# in case the function to be used is already a built-in Python function then we can pass it through a 
# string e.g.:
#     
#     grouped.agg('sum')
# 

# <p> Once we group by a dataframe by two or more column element (this will result in high level layered
#     index). We can aggregate by particular functions (built-in or user defined). Just pass the    
#     functions as list like object within .agg().
#     
# <p> Here we can also give our own defined names for each of these functions, these will then appear as 
#     the names of the columns in the new dataframe. Syntax is implemented as 2-tuple e.g.:
#     
#         grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])

# <p> In case we want to return back the dataframe with a reset index after performing the grouping 
#     we can add an optional argument the as_index = False (or call the dataframe.reset_index in the 
#     next line)

# <h3> General split - apply -combine  </h3>

# In[87]:


frame = pd.DataFrame({'data1': np.random.randn(1000),
        'data2': np.random.randn(1000)}) 


# In[88]:


frame


# In[89]:


quartiles = pd.cut(frame.data1, 4) #let's break the data into 4 buckets of equal 
                                   #length 


# In[90]:


quartiles[:10]


# <p> Therefore, cut returns a categorical object that can in turn be passed in a groupby method
#     to come up with an elementary analysis.

# In[91]:


def get_stats(group):
       return {'min': group.min(), 'max': group.max(),
       'count': group.count(), 'mean': group.mean()} #create a dict that will return a frame with 
                                                     #included values/results of functions 


# In[92]:


grouped = frame.data2.groupby(quartiles).apply(get_stats)


# In[93]:


grouped #we would like to unstack this so that it will look closer to a dataframe


# In[94]:


grouped.unstack() #this will add a higher column layer 


# In[95]:


grouping = pd.qcut(frame.data1, 10, labels = False) #take out the labels to just show the quartiles


# In[96]:


grouped = frame.data1.groupby(grouping).apply(get_stats)


# In[97]:


grouped.unstack() #unstack the Series like object and voila


# <p> Let's come back to the context of missing values, in general we will either drop these values 
#     (usually this is the case where our analysis is not affected by these values) or fill these 
#     missing data with values coming from the frame (usually the mean, median or most frequent value 
#     in case of a categorical variable).

# In[98]:


s = pd.Series(np.random.randn(6))
s[::2] = np.nan #generate missing values in the Series


# In[99]:


s


# In[100]:


s.fillna(s.mean())


# In[101]:


states = ['Ohio', 'New York', 'Vermont', 'Florida',
        'Oregon', 'Nevada', 'California', 'Idaho']
group_key = ['East'] * 4 + ['West'] * 4


# In[102]:


group_key


# In[103]:


data = pd.Series(np.random.randn(8), index=states)
data


# In[104]:


data[['Vermont', 'Nevada', 'Idaho']] = np.nan


# In[105]:


data


# In[106]:


fill_mean = lambda g: g.fillna(g.mean())


# In[107]:


data.groupby(group_key).apply(fill_mean) #we just replace the missing value for west/east city 
                                         #with the mean from for west/east


# In[108]:


data.groupby(group_key).mean()


# In[109]:


fill_values = {'East': 0.5, 'West': -1}
fill_func = lambda g: g.fillna(fill_values[g.name]) #take advantage of the internal
                                                    #attribute g.name of a groupby object?
data.groupby(group_key).apply(fill_func)


#  <h2> Missing Values</h2>

# <p> The usual notation for missing values is NaN. </p>

# <p> Functions we can use include<p\>
#     <ul>
#      <li> dropna() select the data that we would like to drop
#      <li> fillna() fill missing values with a desired values like a mean or the most frequent value
#      <li> isnull() check the entries of the data in the dataframe where we have missing values
#     <\ul>

# <h3> Droping data </h3>

# In[110]:


from numpy import nan as NA 
import pandas as pd
import numpy as np 


# In[111]:


data = pd.Series([1, NA, 3.5, NA, 7])  


# In[112]:


data[data.isnull()] #returns values and indices of the Series where we have missing values 


# In[113]:


data = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA],
     [NA, NA, NA], [NA, 6.5, 3.]])
data


# In[114]:


data.dropna()


# In[115]:


data.dropna(how  = "all") #drops the rows constituting only of missing data 


# In[116]:


data[4] = NA


# In[117]:


data


# In[118]:


data.dropna(how = "all", axis = 1)
data


# In[119]:


df = pd.DataFrame(np.random.randn(7, 3))
df


# In[120]:


df.iloc[:4,1] = NA
df.iloc[:2, 2] = NA
df


# In[121]:


df.dropna(thresh = 2) #select rows with a certain number of observations i.e. columns with two non missing values


# <h3> Filling data </h3>

# In[122]:


df.fillna(0)


# In[123]:


df.fillna({1:0.5, 2:0.7}) #note that fillna returns a new object by default


# In[124]:


df.fillna(0, inplace = True)
df


# In[125]:


df = pd.DataFrame(np.random.randn(6, 3))


# In[126]:


df.iloc[4:, 1] =NA
df.iloc[2:,2] = NA
df


# In[127]:


df.fillna(method = "ffill")
df


# In[128]:


df.fillna(method = "ffill", limit = 2)


# In[129]:


data = pd.Series([1,NA,4,NA, 3.5])
data


# In[130]:


data.fillna(data.mean()) #replace the missing values with the mean value, another candidate is the median


# <h3> Ways to fill missing values </h3>
#             

#  <p>
#     <ul>
#         <li> fill missing values with a desired number, usually the mean or median 
#         <li> method back filling bfill or forward filling ffill 
#         <li> limit, use that in interpolation with the ffill or bfill method
#         <li> remember that the fillna method (as well as the dropna) returns a new object. Inplace = True operates on the initial dataframe
#     <ul\>
#  <p\>

# <h3> Duplicated Values

# In[131]:


data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'],
       'k2': [1, 1, 2, 3, 3, 4, 4]})
data


# In[132]:


data.duplicated()  #checks if a row is duplicate value of another row


# In[133]:


data.drop_duplicates() # simply drop the duplicated rows


# In[134]:


data['v1'] = range(7)


# In[135]:


data.drop_duplicates(['k1']) #select the column on which you  will drop the duplicates
data                         #again notice that the drop_duplicates() creates a new object


# In[136]:


# drop_duplicates() returns the first observed values and cuts down the subsequent duplicates
# we can select to cut down the first occurence of the sequence and return the last values


# In[137]:


data.drop_duplicates(['k1', 'k2'], keep = 'last') # this will cut down the index 5 but will return the index 6 


# <h3> Transforming Data with a function or a mapping <\h3>

# In[138]:


data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
      'Pastrami', 'corned beef', 'Bacon',
      'pastrami', 'honey ham', 'nova lox'],
      'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data


# In[139]:


meat_to_animal = {
'bacon': 'pig',
'pulled pork': 'pig',
'pastrami': 'cow',
'corned beef': 'cow',
'honey ham': 'pig',
'nova lox': 'salmon'
} #consider that we would like to change via mapping the food to the its raw material 


# In[140]:


lower_string = data['food'].str.lower() #make the strings from mixed lowercased/uppercased to lowercased, 
                                        #notice that str.lower() is a Series method


# In[141]:


lower_string


# In[142]:


data['animal']  = lower_string.map(meat_to_animal) # a Series can accept through mapping a dict for changing its elements to the values of the dict, elements and keys should be 1-1.


# In[143]:


data['animal'] = data['food'].map(lambda x: meat_to_animal[x.lower()])
data


# <p> Therefore, we see that map() can transform a subset of data to a format of our desire, however there are
#     easier and more flexible ways go do so, one of them being the replace() function.
# <p\>

# In[144]:


data = pd.Series([1., -999., 2., -999., -1000., 3.])
data


# In[145]:


data.replace(-999, NA)
data


# In[146]:


data.replace([-999,-1000], NA) #replace a list of elements with one value


# In[147]:


data.replace([-999, -1000], [NA, 0]) #replac the elements of a list with another list by 1-1


# In[148]:


data.replace({-999: 0, -1000: NA}) # replacement can also occur via dict
                                  #notice that the method replace() is different from the str.replace() 
                                  # the last one is  a string substitution element-wise


# In[149]:


data = pd.DataFrame(np.arange(12).reshape((3, 4)),
       index=['Ohio', 'Colorado', 'New York'],
       columns=['one', 'two', 'three', 'four'])


# In[150]:


transform  = lambda x: x[:4].upper()


# In[151]:


data.index.map(transform)


# In[152]:


data.rename(index = str.title, columns = str.upper)


# In[153]:


data.rename(index={'Ohio': 'Indiana'},
            columns={'three': 'peekaboo'}, #inplace = True/False
           )


# <h3> Discretization and Binning </h3>

# In[154]:


ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
group_names = ['millenia', 'young_adult', 'middle-age', 'senior']
cats = pd.cut(ages, bins, labels = group_names, right = False, #precision = number [numerical data]
             )
cats


# In[155]:


cats.categories


# In[156]:


cats.codes


# In[157]:


pd.value_counts(cats)


# In[158]:


#instead of bins you can cut into quartiles i.e. 
#pd.cut(data, 4)
#pd.cat(data, [0.2,0.4,0.6,0.8,1.]) create your own percentiles 


# In[159]:


data = pd.DataFrame(np.random.randn(1000, 4))
outliers = data[(np.abs(data) >3).any(1)]  #find the outliers in the entire dataframe 
outliers


# <h3> Permutation and Sampling </h3>

# In[160]:


df = pd.DataFrame(np.arange(20).reshape((5,4)))


# In[161]:


sampler = np.random.permutation(5)
sampler


# In[162]:


df.take(sampler)


# In[163]:


df.sample(n=10 ,replace=True) #replace allows for repetitions


# <h3> Dummy Variables </h3>

# In[170]:


np.random.seed(12345)
values = np.random.rand(10)
values


# In[171]:


bins = [0, 0.2, 0.4, 0.6, 0.8, 1]


# In[172]:


pd.get_dummies(pd.cut(values, bins)) # this will introduce dummy variables in each cateogorization
                                     #1 means existence of the value and 0 means otherwise


# <h3> String Manupulation</h3>

# In[173]:


#split() and strip() can be used together to put strings in a list and then trim any whitespaces 


# In[174]:


val = "a, b, guide" 


# In[175]:


list_val = val.split(",")
list_val


# In[176]:


no_space_ListVal = [x.strip() for x in list_val] # strip will trim any white spaces 
no_space_ListVal


# In[177]:


#check  the directory of functions for string manipulation in mckinney 


# <h3> <i>Regexp </i></h3>

# In[178]:


#regexp can be used for pattern matching, substitution and splitting 


# <h2> Hierarchical Indexing </h2>

# <p> Hierarchical indexing is useful for reshaping data and manipulating group based operations (like pivot tables)
#  <p\>

# In[179]:


import pandas as pd 
import numpy as np


# <h3> Inner indexing  </h3>

# In[180]:


data = pd.Series(np.random.randn(9),
index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
[1, 2, 3, 1, 3, 1, 2, 2, 3]])

data


# In[181]:


data.loc[:, 2] #inner indexing 


# In[182]:


data.unstack() # reshape my Series to a Dataframe


# In[183]:


data.unstack().stack() #turn back to the initial Series


# In[184]:


frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
columns=[['Ohio', 'Ohio', 'Colorado'],
['Green', 'Red', 'Green']])

frame


# In[185]:


frame.index.names = ['key1', 'key2'] # set the names for the multi-index
frame.columns.names = ['State', 'Color'] # set the names for the column 

frame


# <h3> Aggregating values at a certain level </h3>

# In[186]:


frame.sum(level = 'key1')


# In[187]:


frame.sum(level= 'State', axis = 1)


# In[188]:


frame.swaplevel(0,1).sort_index(0) 
#could also be written as frame.swaplevel('key1', 'key2').sort_index(0)


# <h3> Deeper indexing using the dataframe's columns </h3>

# In[189]:


frame = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
'c': ['one', 'one', 'one', 'two', 'two',
'two', 'two'],
'd': [0, 1, 2, 0, 1, 2, 3]})


# In[190]:


frame


# In[191]:


frame2 = frame.set_index(['c','d'])
frame2


# In[192]:


frame.set_index(['c','d'], drop = False) 
#hold the initial columns of the dataframe
frame2.index.names = ['first' ,'second']
#we can now rename the index names so that we do not 
# mess up with the column names
frame2


# In[193]:


frame2.reset_index() 
#return back to the initial object


# <h3> Combining and merging datasets </h3>

# <ul>
#     <li> pandas.merge() merges rows of different dataframes based on index values 
#     <li> pandas.concat() simply 'stacks' together objects along an axis (0,1) 
#     <li> combine_first fills values of one object with values from another object 

# In[194]:


df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
'data1': range(7)})

df1


# In[195]:


df2 = pd.DataFrame({'key': ['a', 'b', 'd'],
'data2': range(3)})

df2


# In[196]:


pd.merge(df1,df1)  #many to one join


# In[197]:


pd.merge(df1, df2, on = "key") #good practice to specify on which column we will merge 


# In[198]:


df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
'data1': range(7)})
df3


# In[199]:


df4 = pd.DataFrame({'rkey': ['a', 'b', 'd'],
'data2': range(3)})

df4


# In[200]:


pd.merge(df3, df4, left_on = 'lkey', right_on = 'rkey') #different keys so we specify each of them 


# <p> Notice that until now merge() does by default an inner join, we need to specify excplicitly 
#     an outer join.

# In[201]:


pd.merge(df1, df2, how = "outer") #outer join, this obviously gives us some NaN values
                                  #in the non overlaping data


# <ul>
#     <li> merge() can have non intuitive results on a many-to-many relationship between two 
#         dataframes, usually in this occasion if three rows appear with the same values in two rows of 
#         second dataframe then the fibal output will be 3x2 = rows.
#     <li> merge() can also be used between overlaping indexes or a mix of index and column values 
#          in this occasion syntax can go like pd.merge(df1, df2, lkey = "...", right_index = "..." how =          ['inner' |'outer'| 'left'])  
#     <li> the .join can be applied by default on indices and in more than two dataframes.
#         Syntax goes like df1.join(df2, how = "...", on = " "). Where the on attribute is used in case
#         we want to join the index of the passed dataframe with the column of the called dataframe.
#        

# <h3> Concat dataframes and series </h3>

# <p> concat() works by default along 0 axis.

# In[202]:


s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])


# In[203]:


pd.concat([s1,s2,s3] )


# In[204]:


pd.concat([s1,s2,s3], axis=1 )


# In[205]:


s4 = pd.concat([s1, s3])
s4


# In[206]:


pd.concat([s1,s4], axis = 1, join= 'inner') #take the intersectio of concatenation 


# In[207]:


df1 = pd.DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],
columns=['one', 'two'])

df2 = pd.DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'],
columns=['three', 'four'])


# In[208]:


pd.concat([df1, df2], axis=1, keys=['level1', 'level2'], join= 'inner')


# In[209]:


# pd.concat({'level1': df1, 'level2': df2}, axis=1) 
# same as
# pd.concat([df1, df2], keys = ['level1, level2'], axis = 1)


# In[210]:


df1 = pd.DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])


# In[211]:


pd.concat([df1,df2], ignore_index= True) #reformating the index


# <h3> Combining data with overlap </h3>

# <p> combine_first() in the case of a dataframe 'patches' missing data column by column in the calling object with data from the object you pass.
#     
# 

# <h3> Reshaping and pivoting  </h3>

# In[212]:


data = pd.DataFrame(np.arange(6).reshape((2, 3)),
index=pd.Index(['Ohio', 'Colorado'], name='state'),
columns=pd.Index(['one', 'two', 'three'],
name='number'))


# In[213]:


data


# In[214]:


result = data.stack()
result    #stack rotates the columns into rows resulting in hierarchical intexed Series 


# In[215]:


result.unstack() #by default the innnermost level will remain unstack


# In[216]:


result.unstack(0) #select which level I would like to unstack


# <p> Unstacking can introduce missing values  

# In[217]:


s1 = pd.Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([4, 5, 6], index=['c', 'd', 'e'])

data2 = pd.concat([s1,s2], keys = ['key1', 'key2'])
data2


# In[218]:


data2.unstack() #innermost level would get the same results as the number of the values 
                #within the innermost level are not the same. 


# In[219]:


data2.unstack().stack() #from here we see that the stack method by default drops values 


# In[220]:


data2.unstack().stack(dropna = False) 


# In[221]:


df = pd.DataFrame({'left': result, 'right': result + 5},
columns=pd.Index(['left', 'right'], name='side'))


# In[222]:


df


# In[223]:


df.unstack(0)


# In[224]:


df.unstack(0).stack(0) #unstacking and stacking at the same level can bring different results

