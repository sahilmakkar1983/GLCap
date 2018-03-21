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


