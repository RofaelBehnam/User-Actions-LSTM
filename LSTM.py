import pandas as pd 
import numpy
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import dump_svmlight_file
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.datasets import dump_svmlight_file 
import itertools
from sklearn.ensemble import RandomForestClassifier 
from sklearn.utils import check_array as check_arrays
from numpy import ones, mean
from keras.utils.np_utils import accuracy
# fix random seed for reproducibility
numpy.random.seed(7)
cols = 1
# read from file
expertsDF = pd.read_json("experts.json")
# choose only interaction events
expertsDF = expertsDF [expertsDF['event'].str.contains("input|click|doubleclick") == True]

# drop all other columns
expertsDF.drop(expertsDF.columns[[0, 1, 2, 3, 6, 7, 8, 9]], axis=1, inplace=True)

# groub by session and put them in a sequence next to session
expertsDF = expertsDF.groupby('session', as_index=False).agg(lambda x: x.tolist())

# drop session column, we don't need it anymore
expertsDF.drop(expertsDF.columns[0], axis=1, inplace=True)
events = set([])
for index, row in expertsDF.iterrows():
    session = row['name'] 
    for x in session:
        events.add(x)
############## 
states = list(events)
samples = []
last = 0
for index, row in expertsDF.iterrows():
    observations = row['name'] 
    n_observations = len(observations)
    # predict a sequence of hidden states based on visible states
    for w in range(0, n_observations-cols):
        sample = []
        for x in range(0, w+cols):
            idx = 0
            for s in range(0, len(states)):
                if observations[x] == states[s]:
                    idx = s;
            sample.append(idx)
        
        if len(sample)==cols:
            samples.append(sample)
            last = index
dataset = samples
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=1000, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))