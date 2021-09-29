# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 20:27:45 2021

@author: Sagar Anil Tiwari
"""




## balancing dataset using over sampling.


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
import os 



os.chdir('C:/Users/Sagar Anil Tiwari/Desktop/Pythonn lectures/DATASETSS')

data = pd.read_excel('data_le.xlsx')
data

data['Status'].value_counts()
data.info()

# partition dataset
x = data.iloc[:,:-1]

y = data.iloc[:,20:]

print(x.shape)
print(y.shape)


data.isnull().sum().any()


count_classes = pd.value_counts(data['Status'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Developed and not developed")

plt.xticks(range(2), LABELS)

plt.xlabel("Status")

plt.ylabel("Frequency")


Developed = data[data['Status']==1]

Notdev = data[data['Status']==0]

print(Developed.shape,Notdev.shape)


from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss


# Implementing Oversampling for Handling Imbalanced 
smk = SMOTETomek(random_state=42)
x_res,y_res=smk.fit_resample(x,y)


x_res.shape,y_res.shape

x_res['Status'].value_counts()


from collections import Counter
print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))



## RandomOverSampler to handle imbalanced data

from imblearn.over_sampling import RandomOverSampler

os =  RandomOverSampler(sampling_strategy = 1)


x_train_res, y_train_res = os.fit_resample(x, y)


x_train_res.shape,y_train_res.shape

x_train_res

y_train_res['Status'].value_counts()



os_us = SMOTETomek(sampling_strategy=1)

x_train_res1, y_train_res1 = os_us.fit_resample(x, y)


x_train_res1.shape,y_train_res1.shape

print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_train_res1)))


####### knn on new dataset

'''
count_classes = pd.value_counts(x_train_res1 , y _train_res1, sort = True)


df = pd.DataFrame(x_train_res1 , y _train_res1)



x_train_res
y_train_res



display(x_train_res,y_train_res)

### concatinating two data

# concatenating df1 and df2 along rows 
vertical_concat = pd.concat([x_train_res,y_train_res], axis=0) 
  '''




#########3concatinating df3 and df4 along row
df = pd.concat([x_train_res,y_train_res], axis=1) # Dataset used for KNN.
  




########## knn new updated dataset


#df = pd.read_excel('Life_Expectancy_KNN_R.xlsx')
#df


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

## standard variable
# Feature scaling.
scaler.fit(df.drop('Status',axis=1))


scaled_features = scaler.transform(df.drop('Status',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()

import seaborn as sns

sns.pairplot(df,hue='Status')


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['Status'],test_size=0.30)



from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)


pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score


print(confusion_matrix(y_test,pred))


print(classification_report(y_test,pred))



# Will take some time

accuracy_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,df_feat,df['Status'])
    accuracy_rate.append(score.mean())


# Will take some time
error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,df_feat,df['Status'],cv=10)
    error_rate.append(1-score.mean())



##plotting


plt.figure(figsize=(10,6))
#plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
  #       markerfacecolor='red', markersize=10)
plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Accuracy Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy Rate')



# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))




## with k = 25
knn = KNeighborsClassifier(n_neighbors = 7)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=25')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

