# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:44:36 2020

@author: u21l12
"""

import pandas as pd
import datetime as dt
import seaborn
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import datetime
import seaborn as sn
import matplotlib.pyplot as plt

df_error_1333446 = pd.read_csv("file:///E:/Device_anomaly_detection/MSS_error_par/Equp_MSS_error/000000000001600967.csv")
#arrange by time
df_error_1333446['measuredAt']=pd.to_datetime(df_error_1333446['measuredAt'])
df_error_1333446=df_error_1333446.sort_values('measuredAt',ascending=True)

#data having value in Z_y_Error
df_error_1333446_1 = df_error_1333446.dropna(subset=['Z_y_Error'])
df_error_1333446_1 = df_error_1333446_1.reset_index(drop = True)
df_error_1333446_1['Z_y_Error'] =df_error_1333446_1['Z_y_Error'].astype(str) #change into string
df_col = list(df_error_1333446.columns)
#df_col.remove('Z_y_Error')

#split by |
s = df_error_1333446_1['Z_y_Error'].str.split('|').apply(pd.Series, 1).stack()
s.index = s.index.droplevel(-1) # to line up with df's index
s.name = 'Z_y_Error'
del df_error_1333446_1['Z_y_Error']
df_error_1333446_1 = df_error_1333446_1.join(s)

df_error_1333446_1['Z_y_Error'] =df_error_1333446_1['Z_y_Error'].astype(float)
df_error_1333446_1['Z_y_Error'] =df_error_1333446_1['Z_y_Error'].astype(str)

df_error_1333446_2= df_error_1333446[~df_error_1333446['Z_y_Error'].notnull()]
df_error_1333446_final = pd.concat([df_error_1333446_1,df_error_1333446_2]).reset_index(drop=True)
df_error_1333446_final=df_error_1333446_final.sort_values('measuredAt',ascending=True)
df_error_1333446_final = df_error_1333446_final.reset_index(drop = True)

#asign flag when error comes
df_error_1333446_final['flag'] = None
df_error_1333446_final['flag'] = np.where((df_error_1333446_final['Z_y_Error'] =="20010.0" ),1,0)

'''
l1=list(df_error_1333446_final[df_error_1333446_final['Z_y_Error'] == '20010.0']['measuredAt'])
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16,10), sharex=True)
pp=df_error_1333446_final.plot('measuredAt','Y_y_RelPressure',ax=axes[0],kind='line')
for i in l1:
    pp.axvline(pd.to_datetime(i), c='r')
pp3=df_error_1333446_final.plot('measuredAt','Y_y_AbsPressure',ax=axes[1],kind='line')
for i in l1:
    pp3.axvline(pd.to_datetime(i), c='r') 
plt.show()
'''
#Define function to take previous data
from pandas import DataFrame
from pandas import concat
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
            """
            Frame a time series as a supervised learning dataset.
            Arguments:
                        data: Sequence of observations as a list or NumPy array.
                        n_in: Number of lag observations as input (X).
                        n_out: Number of observations as output (y).
                        dropnan: Boolean whether or not to drop rows with NaN values.
            Returns:
                        Pandas DataFrame of series framed for supervised learning.
            """
            n_vars = 1 if type(data) is list else data.shape[1]
            df = DataFrame(data)
            cols, names = list(), list()
            # input sequence (t-n, ... t-1)
            for i in range(n_in, 0, -1):
                        cols.append(df.shift(i))
                        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
            # forecast sequence (t, t+1, ... t+n)
            for i in range(0, n_out):
                        cols.append(df.shift(-i))
                        if i == 0:
                                    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                        else:
                                    names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            # put it all together
            agg = concat(cols, axis=1)
            agg.columns = names
            # drop rows with NaN values
            if dropnan:
                        agg.dropna(inplace=True)
            return agg

#---when error occured and continously comes then remove continued error rows (Drop consecutive duplicats)
#data prepared by removing rows having same consicutive value value in flag (20010 in z_y_error)
f = df_error_1333446_final.loc[df_error_1333446_final['flag'].shift(1) != df_error_1333446_final['flag']]
f = f.dropna(subset=['Z_y_Error'])
unq_errorcode = list(pd.unique(f['Z_y_Error']))
f = f[f['Z_y_Error'] == "20010.0"]

f2 = df_error_1333446_final[(~df_error_1333446_final['Z_y_Error'].notnull()) | (df_error_1333446_final['Z_y_Error'] == "21138.0") |(df_error_1333446_final['Z_y_Error'] == "10012.0") |(df_error_1333446_final['Z_y_Error'] == "11115.0")]

df_final = pd.concat([f,f2]).reset_index(drop=True)
df_final=df_final.sort_values('measuredAt',ascending=True)
df_final = df_final.reset_index(drop = True)

l1=list(df_final[df_final['Z_y_Error'] == '20010.0']['measuredAt'])
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16,10), sharex=True)
pp=df_final.plot('measuredAt','Y_y_RelPressure',ax=axes[0],kind='line')
for i in l1:
    pp.axvline(pd.to_datetime(i), c='r')
pp3=df_final.plot('measuredAt','Y_y_AbsPressure',ax=axes[1],kind='line')
for i in l1:
    pp3.axvline(pd.to_datetime(i), c='r')
plt.show()


### taking average over 1min
#channels related to 20010 :Y_y_DiffPressure, Y_y_RelPressure,  Y_y_AbsPressure
df_1min = df_final[['measuredAt','Y_y_DiffPressure', 'Y_y_RelPressure',  'Y_y_AbsPressure' ]]
df_1min =df_1min.reset_index(drop= True)

df_flag_1min = df_final[['measuredAt','flag']]
df_flag_1min =df_flag_1min.reset_index(drop= True)

#
df2_1min = df_1min.resample(rule='1Min', on='measuredAt').mean()
df2_1min = df2_1min.reset_index()

df3_1min = df_flag_1min.resample(rule='1Min', on='measuredAt').max()
df3_1min = df3_1min.drop(['measuredAt'],axis = 1)
df3_1min = df3_1min.reset_index()

#
#merge df2 & df3
merge_df_1min = df3_1min.merge(df2_1min, on ="measuredAt")

### take previous 5 lag values
d1_1min = series_to_supervised(list(merge_df_1min['Y_y_RelPressure']), 5)
d2_1min = series_to_supervised(list(merge_df_1min['Y_y_AbsPressure']), 5)
d3_1min = series_to_supervised(list(merge_df_1min['Y_y_DiffPressure']), 5)
d4_1min = series_to_supervised(list(merge_df_1min['flag']), 5)

for i in range(0,6):
    d1_1min = d1_1min.rename(columns={d1_1min.columns[i]: "Y_y_RelPressure(t-"+ str(5-i)+ ")"})

for i in range(0,6):
    d2_1min = d2_1min.rename(columns={d2_1min.columns[i]: "Y_y_AbsPressure(t-"+ str(5-i)+ ")"})

for i in range(0,6):
    d3_1min = d3_1min.rename(columns={d3_1min.columns[i]: "Y_y_DiffPressure(t-"+ str(5-i)+ ")"})

for i in range(0,6):
    d4_1min = d4_1min.rename(columns={d4_1min.columns[i]: "flag(t-"+ str(5-i)+ ")"})

final_df_1min = pd.concat([d1_1min.drop('Y_y_RelPressure(t-0)', axis=1),d2_1min.drop('Y_y_AbsPressure(t-0)', axis=1),d3_1min.drop('Y_y_DiffPressure(t-0)', axis=1) ,d4_1min[['flag(t-0)']]], axis = 1)
final_df_1min = final_df_1min.reset_index(drop= True)
final_df_1min['Index'] = final_df_1min.index

#Check flag
final_df_1min['flag(t-0)'].value_counts()

### Deep learning model to perform Binary classification
input_col = list(final_df_1min.columns.values)
input_col.remove('flag(t-0)')
input_col.remove('Index')
print(input_col)


X = final_df_1min[input_col]
Y = final_df_1min['flag(t-0)']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_train = X_train.reset_index(drop = True)
y_train = y_train.reset_index(drop = True)
X_test = X_test.reset_index(drop = True)
y_test= y_test.reset_index(drop = True)


y_train.value_counts()
y_test.value_counts()

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score

# Model creation
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(15,)),
    keras.layers.Dense(30, activation=tf.nn.relu),
            keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
             loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=10)
test_loss, test_acc = model.evaluate(X_test, y_test)
test_predict = model.predict(X_test)
test_predict = pd.DataFrame(test_predict)
test_predict['class'] = None
#test_predict['class'] = np.where((test_predict[test_predict.columns[0]] >= 0.05 ),1,0)
test_predict['class'] = np.where((test_predict[test_predict.columns[0]] >= 0.02 ),1,0)
test_predict['Actual class'] = y_test

print(confusion_matrix(y_test, test_predict['class']))
print(accuracy_score(y_test, test_predict['class']))


print('Test accuracy:', test_acc)

a= np.array([[4.02,70.86,62.05,7.0],[2.99,60.30,57.46,6.06]])
print(model.predict(a))










#####################################################
#apply random forest
