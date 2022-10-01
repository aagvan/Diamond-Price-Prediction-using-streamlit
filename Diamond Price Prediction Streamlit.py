#!/usr/bin/env python
# coding: utf-8

# ## Case Study - Diamond Price Prediction

# ### Business Understanding - 
# 
# Diamond is one of the strongest and the most valuable substances produced naturally as a form of carbon. However, unlike gold and silver, determining the price of a diamond is very complex because many features are to be considered for determining its price.
# 
# The value of diamonds depends upon their structure, cut, inclusions (impurity), carats, and many other features. The uses of diamonds are many, such as in industries, as they are effective in cutting, polishing, and drilling. Since diamonds are extremely valuable, they have been traded across different countries for centuries now and this trade only increases with time. They are graded and certified based on the "four Cs", which are color, cut, clarity, and carat. These are the only metrics that are being used to the quality of diamonds and sets the price of the diamond. This metric allows uniform understanding for people across the world to buy diamonds, which allows ease of trade and value for what is purchased.

# ### Task
# 
# #### In this notebook, you will learn:
# 
# 1. How to split the given data into Train and Test ?
# 2. How to perform Data Preparation on -
#    - Categorical Columns - OneHotEncoding and LabelEncoding
#    - Numerical Columns - Standardization and Normalization
# 3. How to build ML models that can predict Price of a Diamond ?

# ### Import the required Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Load the Data

# In[2]:


df = pd.read_csv('diamonds.csv')

df.head()


# price price in US dollars ($326 - \$ 18,823)
# 
# carat weight of the diamond (0.2 - 5.01)
# 
# cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)
# 
# color diamond colour, from J (worst) to D (best)
# 
# clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
# 
# x length in mm (0 - 10.74)
# 
# y width in mm (0 - 58.9)
# 
# z depth in mm (0 - 31.8)
# 
# depth total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43 - 79)
# 
# table width of top of diamond relative to widest point (43 - 95)

# In[3]:


df = df[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z', 'price']]

df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.describe().T


# In[9]:


df.isnull().sum()


# ### Machine Learning Problem
# 
# Build a system which can take features of diamond like carat, cut, color, clarity, x, y, z, etc.. and predicts the price of diamond.

# In[10]:


df[['x','y','z']][(df['z']==0) | (df['y']==0) | (df['z']==0)]


# In[11]:


# Replacing 0 with Null values (np.NaN)

df[['x','y','z']] = df[['x','y','z']].replace(0, np.NaN)


# In[12]:


# Checking Null Values

df.isnull().sum()


# In[13]:


# Dropping Null Values

df.dropna(inplace = True)


# In[14]:


# Rechecking Null Values

df.isnull().sum()


# ## Exploratory Data Analysis

# ### 1. Univariate Analysis

# In[15]:


sns.countplot(x = df['cut'])


# In[16]:


sns.countplot(x = df['color'])


# In[17]:


sns.countplot(x = df['clarity'])


# In[18]:


plt.boxplot(df['carat'])

plt.title('weight of the diamond')
plt.xlabel('weight of the diamond')

plt.show()


# In[19]:


plt.boxplot(df['depth'])

plt.title('depth of the diamond')
plt.xlabel('depth of the diamond')

plt.show()


# In[20]:


plt.boxplot(df['x'])

plt.title('lentgh of diamond')
plt.xlabel('lentgh in mm')

plt.show()


# In[21]:


plt.boxplot(df['y'])

plt.title('width of diamond')
plt.xlabel('widht in mm')

plt.show()


# In[22]:


plt.boxplot(df['z'])

plt.title('depth of diamond')
plt.xlabel('depth in mm')

plt.show()


# ### 2. Bivariate Analysis

# In[23]:


plt.scatter(df['carat'], df['price'])

plt.title('Scatter-plot-btw-Price & weight of the diamond')

plt.xlabel('weight of the diamond')
plt.ylabel('price')

plt.grid()

plt.show()


# In[24]:


plt.scatter(df['x'], df['price'])

plt.title('Scatter-plot-btw-Price & lentgh of the diamond')

plt.xlabel('lentgh of the diamond')
plt.ylabel('price')

plt.grid()

plt.show()


# In[25]:


plt.scatter(df['y'], df['price'])

plt.title('Scatter-plot-btw-Price & width of the diamond')

plt.xlabel('width of the diamond')
plt.ylabel('price')

plt.grid()

plt.show()


# In[26]:


plt.scatter(df['z'], df['price'])

plt.title('Scatter-plot-btw-Price & depth of the diamond')

plt.xlabel('depth of the diamond')
plt.ylabel('price')

plt.grid()

plt.show()


# In[27]:


plt.figure(figsize=(15,8))

sns.heatmap(df.corr(), data = df, annot = True, cmap = 'RdBu_r')


# In[28]:


sns.pairplot(df)


# ### Data Preparation
# - Train Test Split
# - Encoding for Categorical Columns
#   - Ordinal : LabelEncoding or OrdinalEncoding
#   - Nominal : OneHotEncoding or get_dummies
# - Encoding for Numerical Columns
#   - Standardization (z-transformation)
#   
# ### We will be following below mentioned steps:
# 1. Identify the Target Variable and Splitting the Data into train and test
# 2. Separating Categorical and Numerical Columns
# 3. Rescaling Numerical Columns (Standardization or z-transformation)
# 4. Applying OneHotEncoding on Categorical Columns
# 5. Applying Label Encoding on Categorical Columns
# 6. Concatinating the Encoded Categorical Features and Scaled Numerical Features

# ### 1. Identify the Target Variable and Splitting the Data into train and test

# In[29]:


y = df['price']

X = df.drop('price', axis = 'columns')


# In[30]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 10)


# In[31]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# ### 2. Separating Categorical and Numerical Columns

# In[32]:


X_train.dtypes


# In[33]:


X_train_cat = X_train.select_dtypes(include = ['object'])

X_train_cat.head()


# In[34]:


X_train_num = X_train.select_dtypes(include = ['int64', 'float64'])

X_train_num.head()


# ### 3. Rescaling Numerical Columns (Standardization or z-transformation) Applying StandardScaler on Numerical Columns

# In[35]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_num_rescaled = pd.DataFrame(scaler.fit_transform(X_train_num), 
                                    columns = X_train_num.columns, 
                                    index = X_train_num.index)

X_train_num_rescaled.head()


# ### 4. Applying Label Encoding on Categorical Columns applying Label Encoding on Categorical Columns

# In[36]:


from sklearn.preprocessing import OrdinalEncoder

le = OrdinalEncoder()

X_train_cat_le = pd.DataFrame(le.fit_transform(X_train_cat), 
                                    columns = X_train_cat.columns, 
                                    index = X_train_cat.index)

X_train_cat_le.head()


# In[37]:


#X_train_cat_le = pd.DataFrame(index = X_train_cat.index)
#X_train_cat_le.head()


# In[38]:


#X_train_cat.cut.unique()


# In[39]:


#cut_encoder = {'Fair' : 1, 'Good' : 2, 'Very Good' : 3, 'Ideal' : 4, 'Premium' : 5}

#X_train_cat_le['cut'] = X_train_cat['cut'].apply(lambda x : cut_encoder[x])

#X_train_cat_le.head()


# In[40]:


#X_train_cat.color.unique()


# In[41]:


#color_encoder = {'J':1, 'I':2, 'H':3, 'G':4, 'F':5, 'E':6, 'D':7}

#X_train_cat_le['color'] = X_train_cat['color'].apply(lambda x : color_encoder[x])

#X_train_cat_le.head()


# In[42]:


#X_train_cat.clarity.unique()


# In[43]:


#clarity_encoder = {'I1':1, 'SI2':2, 'SI1':3, 'VS2':4, 'VS1':5, 'VVS2':6, 'VVS1':7, 'IF':8}

#X_train_cat_le['clarity'] = X_train_cat['clarity'].apply(lambda x : clarity_encoder[x])

#X_train_cat_le.head()


# ### 4. Concatinating the Encoded Categorical Features and Scaled Numerical Features:

# In[44]:


X_train_transformed = pd.concat([X_train_num_rescaled, X_train_cat_le], axis = 'columns')

X_train_transformed.head()


# ## Same steps of data preprocessing for Test Data

# In[45]:


X_test.dtypes


# In[46]:


X_test_cat = X_test.select_dtypes(include = ['object'])

X_test_cat.head()


# In[47]:


X_test_num = X_test.select_dtypes(include = ['int64', 'float64'])

X_test_num.head()


# In[48]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_test_num_rescaled = pd.DataFrame(scaler.fit_transform(X_test_num), 
                                    columns = X_test_num.columns, 
                                    index = X_test_num.index)

X_test_num_rescaled.head()


# In[49]:


from sklearn.preprocessing import OrdinalEncoder

le = OrdinalEncoder()

X_test_cat_le = pd.DataFrame(le.fit_transform(X_test_cat), 
                                    columns = X_test_cat.columns, 
                                    index = X_test_cat.index)

X_test_cat_le.head()


# In[50]:


#X_test_cat_le = pd.DataFrame(index = X_test_cat.index)

#X_test_cat_le.head()


# In[51]:


#X_test_cat.cut.unique()


# In[52]:


#cut_encoder = {'Fair' : 1, 'Good' : 2, 'Very Good' : 3, 'Ideal' : 4, 'Premium' : 5}

#X_test_cat_le['cut'] = X_test_cat['cut'].apply(lambda x : cut_encoder[x])

#X_test_cat_le.head()


# In[53]:


#X_test_cat.color.unique()


# In[54]:


#color_encoder = {'J':1, 'I':2, 'H':3, 'G':4, 'F':5, 'E':6, 'D':7}

#X_test_cat_le['color'] = X_test_cat['color'].apply(lambda x : color_encoder[x])

#X_test_cat_le.head()


# In[55]:


#X_test_cat.clarity.unique()


# In[56]:


#clarity_encoder = {'I1':1, 'SI2':2, 'SI1':3, 'VS2':4, 'VS1':5, 'VVS2':6, 'VVS1':7, 'IF':8}

#X_test_cat_le['clarity'] = X_test_cat['clarity'].apply(lambda x : clarity_encoder[x])

#X_test_cat_le.head()


# In[57]:


X_test_transformed = pd.concat([X_test_num_rescaled, X_test_cat_le], axis = 'columns')

X_test_transformed.head()


# ## 1. Linear Regression

# In[58]:


from sklearn.linear_model import LinearRegression

LinearRegression = LinearRegression()

LinearRegression.fit(X_train_transformed, y_train)


# In[59]:


y_test_pred = LinearRegression.predict(X_test_transformed)


# In[60]:


temp_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

temp_df.head()


# In[61]:


from sklearn import metrics

print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_test_pred))

print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_test_pred))

print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# ## 2.KNN Regression

# In[62]:


from sklearn.neighbors import KNeighborsRegressor

KNeighborsRegressor = KNeighborsRegressor()

KNeighborsRegressor.fit(X_train_transformed, y_train)


# In[63]:


y_test_pred = KNeighborsRegressor.predict(X_test_transformed)


# In[64]:


temp_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

temp_df.head()


# In[65]:


print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_test_pred))

print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_test_pred))

print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# ## 3. Decision Tree Regression

# In[66]:


from sklearn.tree import DecisionTreeRegressor

DecisionTreeRegressor = DecisionTreeRegressor()

DecisionTreeRegressor.fit(X_train_transformed, y_train)


# In[67]:


y_test_pred = DecisionTreeRegressor.predict(X_test_transformed)


# In[68]:


temp_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

temp_df.head()


# In[69]:


print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_test_pred))

print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_test_pred))

print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# ## 4. Random Forest Regression

# In[70]:


from sklearn.ensemble import RandomForestRegressor
 
RandomForestRegressor = RandomForestRegressor()

RandomForestRegressor.fit(X_train_transformed, y_train)


# In[71]:


y_test_pred = RandomForestRegressor.predict(X_test_transformed)


# In[72]:


temp_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

temp_df.head()


# In[73]:


print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_test_pred))

print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_test_pred))

print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# In[74]:


from sklearn.preprocessing import OrdinalEncoder
lebel_encoder = OrdinalEncoder()


# ### Saving the Model (Serialization)

# In[75]:


from pickle import dump

dump(scaler, open('Desktop/models/standard_scaler.pkl', 'wb'))
dump(le, open('Desktop/models/lebel_encoder.pkl', 'wb'))
dump(LinearRegression, open('Desktop/models/lr_model.pkl', 'wb'))
dump(KNeighborsRegressor, open('Desktop/models/knn_model.pkl', 'wb'))
dump(DecisionTreeRegressor, open('Desktop/models/dt_model.pkl', 'wb'))
dump(RandomForestRegressor, open('Desktop/models/rf_model.pkl', 'wb'))


# ### Loading the Model (Deserialization)

# In[76]:


from pickle import load

scaler = load(open('Desktop/models/standard_scaler.pkl', 'rb'))
le = load(open('Desktop/models/lebel_encoder.pkl', 'rb'))
rf_model = load(open('Desktop/models/rf_model.pkl', 'rb'))


# In[84]:


print('enter diamond details')

cut = input()
color = input()
clarity = input()
carat = float(input())
depth = float(input())
table = float(input())
x = float(input())
y = float(input())
z = float(input())


# In[85]:


num_columns = np.array([carat, depth, table, x, y, z])

le_columns = np.array([cut, color, clarity])


# In[86]:


num_columns = num_columns.reshape(1, -1)

le_columns = le_columns.reshape(1, -1)


# In[87]:


num_columns_transformed = scaler.transform(num_columns)

num_columns_transformed


# In[91]:


le_columns_transformed = le.transform(le_columns)

le_columns_transformed


# In[97]:


query_point = pd.concat([pd.DataFrame(num_columns_transformed), pd.DataFrame(le_columns_transformed)], axis=1)

query_point


# In[99]:


price = rf_model.predict(query_point)

print('price of the diamond is ', price)


# In[102]:


df['price'].loc[17000]


# In[ ]:




