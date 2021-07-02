#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns
sns.set_theme(style="ticks", color_codes=True)


# In[2]:


#load data
df = pd.read_csv('in-vehicle-coupon-recommendation.csv')

df.shape #lets check the dimensionality of the raw data


# # Data Exploration
# Let's take a peek into the data and explore the data and its variables. The dataset is a supervised learning dataset with over 12000 instances and 26 attributes; this mean there is an input variable X and an out variable y.  

# In[3]:


#load the data to understand the attributes and data types
df.head()


# In[4]:


#let's look at the data types
df.dtypes 


# It seems that the data has some few numberical datatypes and the rest are string objects, however all the data can be categorized as being categorical datatypes with a mix of binary and ordinal datatypes.

# In[5]:


#change temperature into a category as its an ordinal datatype
df['temperature']=df['temperature'].astype('category')


# # Cleaning The Data

# In[6]:


#check for empty values
df.info()


# There are some missing values in several columns, and the 'car' variable has only 108 non-null values, more than 99% of the values are NaN. We can just drop it off. These variables are insufficient so its best to remove it completely from the data to avoid inaccuracies in the modeling.

# In[7]:


df["car"].value_counts()


# In[8]:


df.drop('car', inplace=True, axis=1)


# Empty values in categorical data can be removed or replaced with the most frequent value in each column.

# Lets iterate through the pandas table and get all the columns with empty or NaN values, and then for each column the code is going to find the largest variable count and fill the empty values with the corresponding variable with maximum count.

# In[9]:


for x in df.columns[df.isna().any()]:
    df = df.fillna({x: df[x].value_counts().idxmax()})


# In[10]:


#change Object datatypes to Categorical datatypes)

df_obj = df.select_dtypes(include=['object']).copy()

for col in df_obj.columns:
    df[col]=df[col].astype('category')
    
df.dtypes


# In[11]:


#lets do some statistcal analysis
df.describe(include='all')


# In[12]:


df.select_dtypes('int64').nunique()


# From the decription above we can tell that 'toCoupon_GEQ5min' has only one unique variable which won't help much in the encoding of the categorical variables. Therefore, its better to drop that column. 

# In[13]:


df.drop(columns=['toCoupon_GEQ5min'], inplace=True)


# Let's plot the distribution charts of all the categorical datatypes.

# In[14]:


fig, axes = plt.subplots(9, 2, figsize=(20,50))
axes = axes.flatten()

for ax, col in zip(axes, df.select_dtypes('category').columns):
    sns.countplot(y=col, data=df, ax=ax, 
                  palette="ch:.25", order=df[col].value_counts().index);

plt.tight_layout()
plt.show()


# We are going to create feature vectors for our modeling by using the LabelEnconder and OneHotEncoder.

# In[15]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

enc = OneHotEncoder(dtype='int64')

df_cat = df.select_dtypes(include=['category']).copy()
df_int = df.select_dtypes(include=['int64']).copy()

df_enc = pd.DataFrame()
for col in df_cat.columns:
    
    enc_results = enc.fit_transform(df_cat[[col]])

    enc_cat = [col + '_' + str(x) for x in enc.categories_[0]]

    df0 = pd.DataFrame(enc_results.toarray(), columns=enc_cat)

    df_enc = pd.concat([df_enc,df0], axis=1)
    
df_final = pd.concat([df_enc, df_int], axis=1)

df_final


# In[16]:


import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from BOAmodel import *
from collections import defaultdict


# In[17]:


""" parameters """
# The following parameters are recommended to change depending on the size and complexity of the data
N = 2000      # number of rules to be used in SA_patternbased and also the output of generate_rules
Niteration = 500  # number of iterations in each chain
Nchain = 2         # number of chains in the simulated annealing search algorithm

supp = 5           # 5% is a generally good number. The higher this supp, the 'larger' a pattern is
maxlen = 3         # maxmum length of a pattern

# \rho = alpha/(alpha+beta). Make sure \rho is close to one when choosing alpha and beta. 
alpha_1 = 500       # alpha_+
beta_1 = 1          # beta_+
alpha_2 = 500         # alpha_-
beta_2 = 1       # beta_-


# In[ ]:


""" input file """
# # notice that in the example, X is already binary coded. 
# # Data has to be binary coded and the column name shd have the form: attributename_attributevalue
# filepathX = 'tictactoe_X.txt' # input file X
# filepathY = 'tictactoe_Y.txt' # input file Y
# df = read_csv(filepathX,header=0,sep=" ")
# Y = np.loadtxt(open(filepathY,"rb"),delimiter=" ")
df = df_final.iloc[:,:-1].reset_index(drop=True)
Y  = df_final.iloc[:,-1].reset_index(drop=True)

lenY = len(Y)
train_index = sample(range(lenY),int(0.70*lenY))
test_index = [i for i in range(lenY) if i not in train_index]
model = BOA(df.iloc[train_index].reset_index(drop=True), Y.iloc[train_index].reset_index(drop=True))
model.generate_rules(supp, maxlen,N)
model.set_parameters(alpha_1, beta_1, alpha_2, beta_2, None, None)
rules = model.SA_patternbased(Niteration, Nchain, print_message=True)

# test
Yhat = predict(rules, df.iloc[test_index].reset_index(drop=True))
TP,FP,TN,FN = getConfusion(Yhat, Y[test_index].reset_index(drop=True))
tpr = float(TP)/(TP+FN)
fpr = float(FP)/(FP+TN)
print('TP = {}, FP = {}, TN = {}, FN = {} \n accuracy = {}, tpr = {}, fpr = {}'.      format(TP,FP,TN,FN, float(TP+TN)/(TP+TN+FP+FN),tpr,fpr))


# In[ ]:




