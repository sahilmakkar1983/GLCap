import pandas as pd
import gensim
from gensim.models import word2vec
import os
import argparse
import numpy as np
import pickle
import re

max_len = 79

sentences = ["Limited experience in commercial design and build of fuel tank.", "Demonstrated understanding and knowledge of commercial fuel tank requirements.  Experience with business jet fuel tank design and integration", "Wipro stock predicted to go high over year", "Wipro to invest in AI technologies"]

#print("Loading Word2Vec Model")
#wiki_model_name = "en.model"
#wiki_model_path = os.path.join("/datadrive","common_datas","word2vec_wiki","wikiModel")
#model = gensim.models.KeyedVectors.load( os.path.join( wiki_model_path, wiki_model_name ) )
#print("Done")

stack = 6

wts = []

for i in range(stack):
	d = {
		"lstm" : "wts" + str(i+1) + "_512_0.5_tanh_b64_e20_lstm.npy",
		"softmax" : "wts" + str(i+1) + "_512_0.5_tanh_b64_e20_softmax.npy"
	}

#wts.append({"lstm" : "", "dense" : ""})
#wts.append({"lstm" : "", "dense" : ""})

wiki_data_path = os.path.join("/datadrive","common_datas","word2vec_wiki")
indexed_vocab_path = "vocab_index.p"

V_index_dict = pickle.load( open( os.path.join( wiki_data_path, indexed_vocab_path ), "rb" ) )

print(V_index_dict['good'])

#vec_len = len( model["at"] )
vec_len = 1000
def get_data(sentences):
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
	
	from keras.preprocessing import sequence
	data_X = sequence.pad_sequences(data_X, maxlen=max_len)
	return data_X
	
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.layers import Flatten

print("Loading Embeddings")
embedding_path = "wiki_embeddings.npy"
embedding_weights = np.load( os.path.join( wiki_data_path, embedding_path ) )
print("Done")

vocab_size = len(V_index_dict)

def base_model():
	print("Model init")
	model = Sequential()
	model.add(Embedding(vocab_size, vec_len, input_length = max_len, weights = [embedding_weights], trainable=False))
	model.add(LSTM(512, kernel_initializer="normal", dropout=0.5, activation='tanh', name = 'lstm'))
	#model.add(Flatten())
	#model.add(Dense(500,activation='relu'))
	model.add(Dense(3, activation='softmax', name = 'softmax'))
	
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	

#print("Generating Test data from sentences ...")
#test_X = get_data(sentences)
#print(test_X.shape)
#del model
#del V_index_dict 

map = {
	0 : "negative",
	1 : "positive",
	2 : "neutral",
}

#for p, sen in zip(preds, sentences):
#	print( "{} has |{}| sentiment".format( sen, map[np.argmax(p)] ) )

def get_accuracy(y_test, preds):
	import numpy as np
	preds_ = []
	y_ = []
	for p, y_t in zip(preds[:], y_test):
		preds_.append(np.argmax(p))
		y_.append(np.argmax(y_t))

	from sklearn.metrics import accuracy_score
	from sklearn.metrics import confusion_matrix
	print("Accuracy on test data : ", accuracy_score(y_, preds_))

	cmat = confusion_matrix(y_, preds_)
	print("Class accuracies : ", cmat.diagonal()/cmat.sum(axis=1))
	return True

while True:
	key = input("Enter text (q to exit) : ")
	if key == 'q' :
		break
	else :
		test_X = get_data([key])
		preds = model_nn.predict(test_X)
		print( "{} has |{}| sentiment".format( key, map[np.argmax(preds)] ) )
		print(preds)	

import pickle
data = pickle.load( open( "data_for_training_v1.p", "rb" ) )

preds = []
X_test = data["X_test"]
y_test = data["y_test"]

del data

model_nn = base_model()
layer_dict = dict([(layer.name, layer) for layer in model_nn.layers])

for i in range(stack):
	print("Model : ", (i+1))
	from keras import backend as K
	K.clear_session()

	#model_nn = base_model()
	#layer_dict = dict([(layer.name, layer) for layer in model_nn.layers])
	model_nn.get_layer('softmax').set_weights( np.load( wts[i]["softmax"] ) )
	model_nn.get_layer('lstm').set_weights( np.load( wts[i]["lstm"] ) )
	model_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	p = model_nn.predict(X_test[:], batch_size = 64)
	preds.append(p)
	print( "Model {} accuracy : {}".format( i+1, get_accuracy( y_test, p ) ) )
	del p

preds = np.array(preds)
preds = np.mean(preds, axis = 0)

print("Ensemble accuracy : ", get_accuracy(y_test, preds))

#preds = model_nn.predict(X_test[:], batch_size = 32)
#get_accuracy(y_test, preds)   
print("Done!!")
