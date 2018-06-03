import os
import json
import random

import h5py
import keras.layers
import keras.optimizers
import keras.utils
import numpy as np
import pytest
import tensorflow as tf
from typing import Tuple, List, Dict
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout, Dense, Input, Embedding, MaxPooling1D, Flatten
import keras
from keras.layers import Dense, Dropout,Conv2D,MaxPooling2D,Flatten,Conv1D,GlobalMaxPooling1D
from keras.models import Model,Sequential
from keras.layers import RNN,SimpleRNN
import numpy as np
import time



def create_youtube_comment_rnn(vocabulary: List[str],n_outputs: int,train_x,train_y,val_x,val_y):
	MAX_WORDS_IN_SEQ = 100
	EMBED_DIM = 500	
	print(train_x)
	model = Sequential()
	model.add(Embedding(10000,1500, input_length=1,mask_zero=True))
	model.add(SimpleRNN(100,activation='tanh',go_backwards=True))
	model.add(Dense(28,activation='relu'))
	model.add(Dense(38,activation='relu'))
	model.add(Dense(n_outputs,activation='sigmoid'))
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
	dic={'x':None, 'y':None,'epochs':10, 'validation_data':None}
	model.fit(x=train_x, y=train_y,epochs=10, validation_data=(val_x,val_y))

	return [model,dic]



###################################################################
#Data Processing
##################################################################
"""fil=open("2018-E-c-En-train.txt",'r')
labels=[]
file_read=fil.readlines()
train=[]
train_1=np.zeros(len(file_read))
count=0
for i in file_read:
	if(count!=0):
		data=i.split("\t")
		label=data[-11:]
		data=data[1:-11]
		datan=[]
		label[len(label)-1]=label[len(label)-1][0]
		for i in range(len(label)):
			label[i]=int(label[i])
		for i in data:
			n=i.split(" ")
			for j in n:
				datan.append(j)
		#train_1[count]=datan
		train.append(datan)
		labels.append(label)
	count=count+1

vocab={}
count=0
for i in train:
	for j in i:
		if j not in vocab.keys():
			vocab[j]=count
			count=count+1
fill=open("2018-E-c-En-dev.txt",'r')
dev_1=[]
dev=[]
dev_labels=[]
file_read=fill.readlines()
count=0
for i in file_read:
	if(count!=0):
		data=i.split("\t")
		label=data[-11:]
		data=data[1:-11]
		datan=[]
		label[len(label)-1]=label[len(label)-1][0]
		for i in range(len(label)):
			label[i]=int(label[i])
		for i in data:
			n=i.split(" ")
			for j in n:
				datan.append(j)
				dev_1.append(j)
		dev.append(datan)
		dev_labels.append(label)
	count=count+1

create_youtube_comment_rnn(vocab.keys(),11,np.array(train),np.array(labels),np.array(dev),np.array(dev_labels))
#print(np.array(train))"""







