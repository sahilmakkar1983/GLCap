import pandas as pd
import numpy as np

import gensim
from gensim.models import word2vec

import os
import argparse
import pickle
import re

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.utils import np_utils

import gc

import commonSentimentalModel

def transformData(sentences):
	data_X = []
	sentences = [re.sub('(?<! )(?=[.,"!?()])|(?<=[.,"!?()])(?! )', r' ', s) for s in sentences]
	for sen in sentences:
		vec = []
		for index, word in enumerate(sen.split()[:max_len]):
			#vec[index] = V_index_dict[word] if word in V_index_dict.keys() else -1
			if word in V_index_dict.keys():
				vec.append(V_index_dict[word])
			else :
				vec.append(0)
		data_X.append(vec)
	
	data_X = np.array(data_X)
	
	data_X = sequence.pad_sequences(data_X, maxlen=max_len)
	return data_X
	
def baseModel():
	print("Model init")
	model = Sequential()
	model.add(Embedding(vocab_size, vec_len, input_length = max_len, weights = [embedding_weights]))
	model.add(LSTM(512, kernel_initializer="normal", dropout=0.5, activation='tanh', name = 'lstm'))
	#model.add(LSTM(256, kernel_initializer="normal", dropout=0.5, activation='tanh', name = 'lstm'))
        #model.add(Flatten())
        #model.add(Dense(500,activation='relu'))
	model.add(Dense(256, kernel_initializer='uniform', activation='relu',name='dense1')) 
	model.add(Dense(128, kernel_initializer='uniform', activation='relu',name='dense2')) 
	model.add(Dense(4, activation='softmax', name = 'softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def loadWeightsToBaseModel(model_nn):
	print(model_nn.summary())

	layer_dict = dict([(layer.name, layer) for layer in model_nn.layers])

	model_nn.get_layer('softmax').set_weights( np.load(w_dense) )
	model_nn.get_layer('dense1').set_weights( np.load(w_dense1) )
	model_nn.get_layer('dense2').set_weights( np.load(w_dense2) )
	model_nn.get_layer('lstm').set_weights( np.load(w_lstm) )

	model_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#for p, sen in zip(preds, sentences):
#	print( "{} has |{}| sentiment".format( sen, map[np.argmax(p)] ) )

def getAccuracy(y_test, preds):
	preds_ = []
	y_ = []
	for p, y_t in zip(preds[:], y_test):
		preds_.append(np.argmax(p))
		y_.append(np.argmax(y_t))

	print("Accuracy on test data : ", accuracy_score(y_, preds_))

	cmat = confusion_matrix(y_, preds_)
	print("Class accuracies : ", cmat.diagonal()/cmat.sum(axis=1))
	return True

def copyFileToSSDPath(ssdPath, filePath):
	if os.path.exists(filePath) == False:
		os.system('cp '+ filePath + ' ' + ssdPath)
	return os.path.join(os.path.basename(ssdPath), filePath)


if __name__ == "__main__":
	max_len = 79

	#import os.path
	import os
	ssdPath = '/mnt'
		
	sentences = ["Limited experience in commercial design and build of fuel tank.", "Demonstrated understanding and knowledge of commercial fuel tank requirements.  Experience with business jet fuel tank design and integration"]

	print("Loading Word2Vec Model")
	wiki_model_name = "en.model"
	wiki_model_path = os.path.join("/datadrive","common_datas","word2vec_wiki","wikiModel")
	wiki_model_file = os.path.join( wiki_model_path, wiki_model_name)
	wiki_model_file = copyFileToSSDPath(ssdPath,wiki_model_file)
	model = gensim.models.KeyedVectors.load( wiki_model_file)
	# Linux specific code, change it later
		
	print("Done")

	w_lstm = "wts6_512_0.5_tanh_b8_e10_lstm.npy"
	w_dense1 = "wts6_512_0.5_tanh_b8_e10_dense1.npy"
	w_dense2 = "wts6_512_0.5_tanh_b8_e10_dense2.npy"
	w_dense = "wts6_512_0.5_tanh_b8_e10_softmax.npy"

	wiki_data_path = os.path.join("/datadrive","common_datas","word2vec_wiki")
	indexed_vocab_path = "vocab_index.p"

	wiki_vocab_file = os.path.join( wiki_data_path, indexed_vocab_path)
	wiki_vocab_file = copyFileToSSDPath(ssdPath,wiki_vocab_file)
	V_index_dict = pickle.load( open( wiki_vocab_file, "rb" ) )

	#print(V_index_dict['good'])

	embedding_path = "wiki_embeddings.npy"
	wiki_embed_file = os.path.join( wiki_data_path, embedding_path)
	wiki_embed_file = copyFileToSSDPath(ssdPath,wiki_embed_file)
	embedding_weights = np.load( wiki_embed_file)
	vec_len = len( model["at"] )

	vocab_size = len(V_index_dict)
	vec_len = len( model["at"] )


	gc.collect()
	#model_nn = baseModel(500,.5,'tanh')
	model_nn = baseModel()


	loadWeightsToBaseModel(model_nn)
	map = {
		0 : "negative",
		1 : "positive",
		2 : "neutral",
	}

	while True:
		key = input("Enter text (q to exit) : ")
		if key == 'q' :
			break
		else :
			testX = transformData([key])
			preds = model_nn.predict(testX)
			print( "{} has |{}| sentiment".format( key, map[np.argmax(preds)] ) )
			print(preds)	


