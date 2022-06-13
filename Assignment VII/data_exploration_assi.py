# -*- coding: utf-8 -*-
"""Data Exploration assi.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d2QxYn7sZOnp73qyLosV9Z3AOmyhNtcn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("drive/MyDrive/covid_data.csv")

df.head()

df.info()

df.select_dtypes(include="object").isnull().sum()

df.describe(include="number")

df.describe(include="object")

columns = ['new_cases','location','total_deaths','female_smokers','male_smokers','handwashing_facilities','hospital_beds_per_thousand']

df[columns]

df[columns].isnull().sum()

df.select_dtypes(include = np.number).corr()

import seaborn as sns

sns.heatmap(df.select_dtypes(include = np.number), vmin=-1, vmax=1, center=0)

plt.show()

location_df = df[df['location'].isin(["Nepal","China","India"])]

sns.barplot(x="location",y="total_deaths",data =location_df)
plt.show

location_df = location_df[columns]

location_df

plt.figure()
sns.barplot(x="location",y="hospital_beds_per_thousand",data =location_df)

numeric_col = location_df.select_dtypes('number').columns
n = len(numeric_col)

fig,ax = plt.subplots(n,1, figsize=(6,6*5), sharex=True)

for i,col in enumerate(numeric_col):
  plt.sca(ax[i])
  sns.countplot(x=col, data=location_df)

sns.boxplot(x="new_cases", data=location_df)

sns.boxplot(x="total_deaths", data=location_df)

fig = plt.figure(figsize = (15,15))
ax = fig.gca()
location_df.select_dtypes(include = np.number).hist(ax = ax)
plt.show()

selected_df = df[[ 'new_cases', 'total_deaths', 'female_smokers', 'male_smokers', 'handwashing_facilities' ,'hospital_beds_per_thousand']]

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

si = SimpleImputer(strategy="median")
selected_np = si.fit_transform(selected_df)

selected_df = pd.DataFrame(data=selected_np, columns=['new_cases','total_deaths', 'female_smokers', 'male_smokers', 'handwashing_facilities' ,'hospital_beds_per_thousand'])

selected_df.isnull().sum()

X = selected_df.drop('new_cases', axis=1)
y = selected_df['new_cases']

X.info()

len(y)

X.shape

st = StandardScaler()
X_std = st.fit_transform(X)

X_std.shape

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size =42, random_state=42)

from sklearn.linear_model import LinearRegression, ElasticNet

lr = LinearRegression()
en = ElasticNet()

lr.fit(X_train, y_train)
en.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_pred_en = en.predict(X_test)

from sklearn.metrics import mean_absolute_error as mse

rmse_lr = mse(y_test, y_pred_lr)**(1/2)
rmse_en = mse(y_test, y_pred_en)**(1/2)

print(rmse_lr, rmse_en)

print(f"Coefficients: {lr.coef_} \n Intercept: {lr.intercept_}" )

print(f"Coefficients: {en.coef_} \n Intercept: {en.intercept_}" )
