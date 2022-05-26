#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis of Campaign Sales Dataset
# 
# In this Assignment, We have used following libraries:
# 
# 1. Pandas
# 2. Matplotlib
# 3. Seaborn

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# # Reading the Dataset

# In[2]:


df = pd.read_csv("AddedFeatures_campaign_sale.csv")


# # Descriptive Statistics of The dataset
# 
# 
# Following are the statistics used in the datasets:
# 
# `.head()` -  For Viewing 5 rows in all columns<br>
# `.info()` - For Getting Descriptions of Null Objects<br>
# `.describe()` - For Getting *Mean, Standard Deviation and other statistics* for categorical and numerical values.

# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe(include = ['O'])


# In[6]:


df.describe(include=[np.number])


# ### Looking out Shape, Columns and Index of the dataset,
# ### Also, looking for Numerical, Binary and Categorical Columns 

# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


df.index


# In[10]:


# Looking for categorical and Numerical features

print("Categorical Features of dataset:"+ str(len([i for i in df.columns if df[i].dtype == object])))
print("Floating Type Numerical Features of dataset:"+ str(len([i for i in df.columns if df[i].dtype == float])))
print("Integer Type Numerical Fetaures of dataset:"+str(len([i for i in df.columns if df[i].dtype == np.int64()])))


# # Graphical Visualization
# 
# Following Visualization is done below:
# 
# 1. Correlation Matrix with Heatmap (Using Seaborn)
# 2. Bar Plot for Binary and Categorical Columns.
# 3. Histogram for Numerical Datas.
# 4. Scatter plot for Number of Prior Transaction and Days Since Last Transaction.
# 5. Boxplot for numerical Columns.
# 6. Pie Plot for categorical columns.
# 

# In[11]:


df.select_dtypes(include = np.number).corr()


# In[12]:


sns.heatmap(df.select_dtypes(include = np.number).corr(), 
            vmin=-1, vmax=1, center=0)


# In[13]:


plt.figure(figsize = (3,3))
for category in df.select_dtypes(include = ["O", bool]):
    df[category].value_counts().plot(kind = "bar")
    plt.show()


# In[ ]:





# In[14]:


fig = plt.figure(figsize = (15,15))
ax = fig.gca()
df.select_dtypes(include = np.number).hist(ax = ax)
plt.show()


# In[15]:


plt.scatter(df['Number of Prior Year Transactions'], df['Days_since_last_transaction'])
plt.xlabel("previous year transaction")
plt.ylabel("Last Transaction Days")
plt.show()


# In[ ]:





# In[16]:


fig = plt.figure(figsize = (5,5))
plt.tight_layout()
for category in df.select_dtypes(include = ["O"] ):
    df[category].value_counts().plot(kind='pie')
    plt.show()


# # Data Quality Verification
# 
# This Segment looks out for Data Quality.
# 
# 1. Outlier Data (Shown in Box Plot)
# 2. Missing Data
# 3. Imbalance Data

# In[17]:


fig = plt.figure(figsize = (1,1))


for category in df.select_dtypes(include = [np.number] ):
#     print(type(category))
    df.boxplot(column = category)
    plt.show()


# In[31]:


pd.isnull(df).sum()


# In[ ]:




