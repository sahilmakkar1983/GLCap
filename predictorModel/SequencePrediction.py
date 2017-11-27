import os
import pickle
import argparse

import gensim
from gensim.models import word2vec

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import nltk
from nltk.tokenize import word_tokenize


from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.utils import to_categorical
#x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
from keras import backend as K
from keras.callbacks import EarlyStopping


def sliding_window(df,window_size,window_stride,input_features=None, output_features=None):
    #print(df.columns)
    #print(df[input_features])
    #print(df[input_features].values)
    allData = df[input_features].values.tolist()
    if output_features:
        outputData = [i[0] for i in df[output_features].values.tolist()]
    #print(outputData)
    myArray = [[]]
    outputDataArray = [] 
    start=0
    for i in range(0,len(allData),window_stride):
        #print(allData[i:window_size+i])
        #print(i)
        if i == 0:
            myArray = [allData[i:window_size+i]]
            if output_features != None:
                #outputDataArray.append(outputData[window_size+i][0])
                outputDataArray=[outputData[i+window_size-1]]
                #print(1,type(outputDataArray))
        else:
            myArray.append(allData[i:window_size+i])
            
            if window_size+i >= len(allData):
                if output_features != None:
                    #print(2,type(outputDataArray))
                    outputDataArray.append(outputData[len(allData)-1])
                break
            if output_features != None:
                    #outputDataArray.append(outputData[window_size+i][0])
                    #print(3,type(outputDataArray))
                    outputDataArray.append(outputData[i+window_size-1])

    if output_features == None:
            return (np.array(myArray))
    #print(outputDataArray)
    return (np.array(myArray), outputDataArray)


# In[76]:


def fileProcessor(csvFile,stockName):
    dateparse = lambda x: pd.datetime.strptime(x, '%d-%m-%Y')
    df = pd.read_csv(csvFile, parse_dates=['date'],date_parser=dateparse)
    #df = pd.read_csv("/datadrive/Sahil/code/GL/fewTrails/Datasets/GE.csv", parse_dates=['date'],date_parser=dateparse)


    df["date"]  = pd.to_datetime(df.date)
    type(df["date"].iloc[0])
    df = df.sort_values(by="date")

    df['close_delta'] = 0
    df['close_direction'] = 0
    df['Stock_name'] = stockName
    for index in range(0,df.shape[0]):
        #print(index,df.iloc[index]['close'])
        if index == 0:
            df['close_delta'].iloc[index] =  (float(0))
            df['close_direction'].iloc[index] =  1
        #elif index <= df.shape[0]-1:
        else:
            df['close_delta'].iloc[index] = ((df['close'].iloc[index] - df['close'].iloc[index-1])/df['close'].iloc[index-1])*100
            if df['close_delta'].iloc[index] >= 1:
                df['close_direction'].iloc[index] = 2
            elif df['close_delta'].iloc[index] <= -1:
                df['close_direction'].iloc[index] = 0
            else:
                df['close_direction'].iloc[index] = 1
    return (df)

fileCount = 0
reParseFile = True
if reParseFile == True:
	for root, dirs, files in os.walk("/datadrive/Sahil/code/GL/fewTrails/Datasets"):
		for file in files:
			if file.endswith(".csv"):
				print(os.path.join(root, file))
				if fileCount == 0:
					df = fileProcessor(os.path.join(root, file),file.split('.')[0])
				else:
					df = df.append(fileProcessor(os.path.join(root, file),file.split('.')[0]))
				fileCount = fileCount + 1
	df.to_csv("/datadrive/Sahil/code/GL/fewTrails/Datasets/ALL_clean.csv")
else:
	df = df.read_csv("/datadrive/Sahil/code/GL/fewTrails/Datasets/ALL_clean.csv")
 

    
df.shape[0]


dfdffd


df.head(15)


# In[50]:

import pandas as pd
import numpy as np
# Get some time series data
#df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/timeseries.csv")
#df['output']=[1,0,1,0,1,0,1,0,1,0,1]


# In[58]:

input_cols = ['curr_ratio','tot_debt_tot_equity', 'oper_profit_margin','asset_turn','ret_equity','sentiment']
mydf = df
x_train,y_train=sliding_window(mydf,5,1,input_cols,['close_direction'])


# In[59]:

x_train


# In[60]:

y_train


# In[61]:

y_train = np.array(y_train)
y_train = np.array(y_train).reshape((-1, 1))
#y_train = to_categorical(y_train)

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_train = onehot_encoder.fit_transform(y_train)


# In[62]:

#onehot_encoded[0:30]


# In[63]:

y_train


# In[64]:

model = Sequential()
# input_shape = number of time-steps, number-of-features
model.add(LSTM(128,input_shape=(5,len(input_cols)),
               activation='sigmoid', 
               inner_activation='hard_sigmoid', 
               return_sequences=True))
model.add(LSTM(128, activation='tanh', recurrent_activation='hard_sigmoid'))
#model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
#model.add(TimeDistributedDense(11))
#model.add(Dense(128))
model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
model.add(Dense(output_dim=3, kernel_initializer='uniform', activation='sigmoid'))
#model.add(Activation('sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


# In[65]:

model.summary()


# In[66]:

print('Train...')
model.fit(x_train, y_train,
          batch_size=1,
          epochs=5,
          validation_data=(x_train, y_train))
#score, acc = 


# In[68]:

score = model.evaluate(x_train, y_train,batch_size=1)
print()
print('Test score:', score)
print('Test accuracy:', confusion_matrix(y_train,model.predict(x_train)))


# In[71]:

#model.predict(x_train)
a = model.predict_classes(x_train)


# In[72]:

b = set(a)


# In[73]:

b


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

ALL


# In[ ]:

a=[[[]]]


# In[ ]:

a[0] = ALL[0:5]


# In[ ]:

a.append(ALL[1:6])


# In[ ]:

a


# In[ ]:

np.array(a)


# In[ ]:

len(ALL)


# In[ ]:



