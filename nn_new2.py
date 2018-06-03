import keras

from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential

from keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout, Dense, Input, Embedding, MaxPooling1D, Flatten,Bidirectional

from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.layers import RNN,SimpleRNN
from sklearn.metrics import jaccard_similarity_score

import os

from sklearn.metrics  import roc_auc_score

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import re
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.metrics import categorical_accuracy,binary_accuracy

from sklearn.model_selection import train_test_split

###########################################################################
#Training Data

tweets=pd.read_csv('2018-E-c-En-train.txt',sep="\t")
data=tweets['Tweet']
tweets['Tweet']= tweets['Tweet'].apply(lambda x: x.lower())
tweets['Tweet']= tweets['Tweet'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

max_fatures = 2500
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(tweets['Tweet'].values)
X = tokenizer.texts_to_sequences(tweets['Tweet'].values)
X = pad_sequences(X,maxlen=31)

embed_dim = 500

model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1],trainable=True))

model.add(Bidirectional(LSTM(100,activation='tanh',dropout=0.5,recurrent_dropout=0.5,go_backwards=False, return_sequences=True)))
model.add(Bidirectional(LSTM(110,activation='tanh',dropout=0.5,recurrent_dropout=0.5,go_backwards=False,return_sequences=False)))

model.add(Dense(132,activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(164,activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(11,activation='sigmoid'))
model.compile(loss ='categorical_crossentropy', optimizer='adam',metrics = [categorical_accuracy])
early_stop = EarlyStopping(monitor='loss', min_delta=0.01, patience=5, verbose=1, mode='min')
checkpointer = ModelCheckpoint(filepath='weights.hdf5', monitor='val_categorical_accuracy',verbose=1, save_best_only=True)
Y = tweets.loc[:,['anger','anticipation','disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust']].values
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size = 0.20, random_state = 42)
batch_size =512
history = model.fit(X_train,Y_train, epochs = 25, callbacks=[checkpointer],batch_size=batch_size, validation_data=(X_val, Y_val))
#######################################################################################################################################
#validating
tweets_1=pd.read_csv('2018-E-c-En-dev.txt',sep="\t")
data_1=tweets_1['Tweet']
tweets_1['Tweet']= tweets_1['Tweet'].apply(lambda x: x.lower())
tweets_1['Tweet']= tweets_1['Tweet'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

max_fatures = 1000
#tokenizer = Tokenizer(num_words=max_fatures, split=' ')
#tokenizer.fit_on_texts(tweets_1['Tweet'].values)
X_1 = tokenizer.texts_to_sequences(tweets_1['Tweet'].values)
X_1 = pad_sequences(X_1,maxlen=31)
Y_1 = tweets_1.loc[:,['anger','anticipation','disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust']].values
X_test=X_1
Y_test=Y_1
model.load_weights('weights.hdf5')
score_v,acc_v=model.evaluate(X_val, Y_val, verbose = 1, batch_size = batch_size)
score,acc = model.evaluate(X_test, Y_test, verbose = 1, batch_size = batch_size)
y_test=model.predict(X_test, verbose = 1, batch_size = batch_size)
predicted=np.zeros(y_test.shape)#(len(y_test),11))
for i in range(len(y_test)):
	for j in range(len(y_test[i])):
			if(y_test[i][j]>0.5):
				predicted[i][j]=1
			else:
				predicted[i][j]=0
##############################################################################################################################################
#Writing into  File for validataion
#anger	anticipation	disgust	fear	joy	love	optimism	pessimism	sadness	surprise	trust
tweets_2=pd.read_csv('2018-E-c-En-dev.txt',sep="\t")
data_uplo=open('E-C_en_pred.txt','w')
data_uplo.write("ID")
data_uplo.write("\t")
data_uplo.write("Tweet")
data_uplo.write("\t")
data_uplo.write("anger")
data_uplo.write("\t")
data_uplo.write("anticipation")
data_uplo.write("\t")
data_uplo.write("disgust")
data_uplo.write("\t")
data_uplo.write("fear")
data_uplo.write("\t")
data_uplo.write("joy")
data_uplo.write("\t")
data_uplo.write("love")
data_uplo.write("\t")
data_uplo.write("optimism")
data_uplo.write("\t")
data_uplo.write("pessimism")
data_uplo.write("\t")
data_uplo.write("sadness")
data_uplo.write("\t")
data_uplo.write("surprise")
data_uplo.write("\t")
data_uplo.write("trust")
data_uplo.write("\t")
#data_uplo.write("Tweet")
data_uplo.write("\n")
uplo=[]
for i in range(len(tweets_2['Tweet'])):
		a=[]
		a.append(tweets_2['Tweet'][i])
		a.append(predicted[i])
		uplo.append(a)
		data_uplo.write(tweets_2['ID'][i])
		data_uplo.write("\t")
		data_uplo.write(tweets_2['Tweet'][i])
		data_uplo.write("\t")
		for j in range(len(predicted[i])):
			data_uplo.write(str(int(predicted[i][j])))
			if(j!=10):		
				data_uplo.write("\t")
		data_uplo.write('\n')
print(jaccard_similarity_score(Y_test,predicted))
###########################################################################################################################
#Testing
tweets_1=pd.read_csv('2018-E-c-En-test.txt',sep="\t")
data_1=tweets_1['Tweet']
tweets_1['Tweet']= tweets_1['Tweet'].apply(lambda x: x.lower())
tweets_1['Tweet']= tweets_1['Tweet'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

max_fatures = 1000
#tokenizer = Tokenizer(num_words=max_fatures, split=' ')
#tokenizer.fit_on_texts(tweets_1['Tweet'].values)
X_1 = tokenizer.texts_to_sequences(tweets_1['Tweet'].values)
X_1 = pad_sequences(X_1,maxlen=31)
#Y_1 = tweets_1.loc[:,['anger','anticipation','disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust']].values
X_test=X_1
#Y_test=Y_1
model.load_weights('weights.hdf5')
#score_v,acc_v=model.evaluate(X_val, Y_val, verbose = 1, batch_size = batch_size)
#score,acc = model.evaluate(X_test, Y_test, verbose = 1, batch_size = batch_size)
y_test=model.predict(X_test, verbose = 1, batch_size = batch_size)
predicted=np.zeros(y_test.shape)#(len(y_test),11))
for i in range(len(y_test)):
	for j in range(len(y_test[i])):
			if(y_test[i][j]>0.5):
				predicted[i][j]=1
			else:
				predicted[i][j]=0
################################################################################################################################
#Writing into File for testing
#anger	anticipation	disgust	fear	joy	love	optimism	pessimism	sadness	surprise	trust
tweets_2=pd.read_csv('2018-E-c-En-test.txt',sep="\t")
data_uplo=open('E-C_en_pred_1.txt','w')
data_uplo.write("ID")
data_uplo.write("\t")
data_uplo.write("Tweet")
data_uplo.write("\t")
data_uplo.write("anger")
data_uplo.write("\t")
data_uplo.write("anticipation")
data_uplo.write("\t")
data_uplo.write("disgust")
data_uplo.write("\t")
data_uplo.write("fear")
data_uplo.write("\t")
data_uplo.write("joy")
data_uplo.write("\t")
data_uplo.write("love")
data_uplo.write("\t")
data_uplo.write("optimism")
data_uplo.write("\t")
data_uplo.write("pessimism")
data_uplo.write("\t")
data_uplo.write("sadness")
data_uplo.write("\t")
data_uplo.write("surprise")
data_uplo.write("\t")
data_uplo.write("trust")
data_uplo.write("\t")
#data_uplo.write("Tweet")
data_uplo.write("\n")
uplo=[]
for i in range(len(tweets_2['Tweet'])):
		a=[]
		a.append(tweets_2['Tweet'][i])
		a.append(predicted[i])
		uplo.append(a)
		data_uplo.write(tweets_2['ID'][i])
		data_uplo.write("\t")
		data_uplo.write(tweets_2['Tweet'][i])
		data_uplo.write("\t")
		for j in range(len(predicted[i])):
			data_uplo.write(str(int(predicted[i][j])))
			if(j!=10):		
				data_uplo.write("\t")
		data_uplo.write('\n')
