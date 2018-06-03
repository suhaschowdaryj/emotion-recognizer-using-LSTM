import keras

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D

from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.layers import RNN,SimpleRNN

import os

from sklearn.metrics  import roc_auc_score

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import re
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout, Dense, Input, Embedding, MaxPooling1D, Flatten

from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy

tweets=pd.read_csv('2018-E-c-En-train.txt',sep="\t")
data=tweets['Tweet']
tweets['Tweet']= tweets['Tweet'].apply(lambda x: x.lower())
tweets['Tweet']= tweets['Tweet'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

max_fatures = 500
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(tweets['Tweet'].values)
X = tokenizer.texts_to_sequences(tweets['Tweet'].values)
X = pad_sequences(X)

embed_dim = 100

lstm_out = 300

model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))

#model.add(Dropout(0.5))

model.add(Conv1D(132, kernel_size=2,strides=2, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Conv1D(264, kernel_size=2,strides=2, activation='relu',padding='same'))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Conv1D(428, kernel_size=2,strides=2, activation='relu',padding='same'))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Conv1D(856, kernel_size=2,strides=2, activation='relu',padding='same'))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Conv1D(512, kernel_size=2,strides=1, activation='relu',padding='same'))
#model.add(GlobalMaxPooling1D())
model.add(Flatten())
model.add(Dense(132, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(264, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(528, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(11, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='sgd',metrics = [categorical_accuracy])
early_stop = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=20, verbose=1, mode='min')
#print(model.summary())
Y = tweets.loc[:,['anger','anticipation','disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust']].values
#print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
#print(X_train.shape,Y_train.shape)
#print(X_test.shape,Y_test.shape)
X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train, test_size = 0.20, random_state = 42)

batch_size =512

history = model.fit(X_train,Y_train, epochs = 50, callbacks=[early_stop],batch_size=batch_size, validation_data=(X_val, Y_val))
