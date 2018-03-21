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
import re


import nltk
from nltk.tokenize import word_tokenize


from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.models import Sequential

from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import EarlyStopping

import commonSentimentalModel

def getEmbeddings(vocab_size, vec_len):
    if embedding_path in os.listdir(wiki_data_path):
        print("Loading Embeddings")
        embedding_weights = np.load( os.path.join( wiki_data_path, embedding_path ) )
    else :
        print("Generating embeddings")
        embedding_weights = np.zeros( (  vocab_size, vec_len ) )
        for word, index in V_index_dict.items():
            embedding_weights[index,:] = model[word] if word in model.wv.vocab else [0]*vec_len
        
        np.save( os.path.join( wiki_data_path, "wiki_embeddings", embedding_weights) )
    
    return embedding_weights

def getIndexedDict(model):
    if indexed_vocab_path not in os.listdir(wiki_data_path):
        print("Generating vocab with index")
        vocab = model.wv.vocab
        V_ = set(vocab)
        
        print( "All words : {}".format( len(vocab) ))
        print( "Vocab Size : {}".format( len(V_) ) )

        V_index_dict = {}
        for index, word in enumerate(V_):
            V_index_dict[word] = index

        pickle.dump( V_index_dict, open( os.path.join( wiki_data_path, indexed_vocab_path ), 'wb') )
    else :
        print("Loading Vocab")
        V_index_dict = pickle.load( open( os.path.join( wiki_data_path, indexed_vocab_path ), "rb" ) )

    return V_index_dict

def removeURLFromText(text):
    #print(str(text))
    text = re.sub(r'b"*', '', str(text), flags=re.MULTILINE)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', str(text), flags=re.MULTILINE)
    text = re.sub(r'http?:\/\/.*[\r\n]*', '', str(text), flags=re.MULTILINE)
    #text = re.sub(r'"', '', str(text), flags=re.MULTILINE)
    text = re.sub(r'^"|"$', '', text)
    text = re.sub(r'\'', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ]',r'',text)
    return (text)

def parseExcelFileWithMultipleSheetsAndCombine(excelFilePath,SheetsToParse):
    df = pd.ExcelFile(excelFilePath,encoding="utf-8")

    masterDf = df.parse("All")

    for index,sheet_name in enumerate(SheetsToParse):
        if index == 0:
            WholeDf = df.parse(sheet_name)
            WholeDf["STOCK"]= sheet_name
        else:
            tempDf = df.parse(sheet_name)
            tempDf["STOCK"]= sheet_name
            WholeDf= WholeDf.append(tempDf) #read a specific sheet to DataFrame

    WholeDf['text'] = WholeDf['text'].apply(lambda x: removeURLFromText(x))
    WholeDfNonNULL =WholeDf[WholeDf.Rating == WholeDf.Rating]
    
    return (WholeDfNonNULL)


def prepareData():
    if wiki_model_name in os.listdir(wiki_model_path):
        model = gensim.models.KeyedVectors.load( os.path.join( wiki_model_path, wiki_model_name ) )
    else :
        print("Word2vec model not found in {}".format(wiki_model_path))

    vec_len = len(model['a'])
    print( "Word2vec Vector length {}".format( vec_len ) )

    SheetsToParse=['AAPL',
               'MSFT',
               'GE',
               'IBM',
               'DIS',
               'PG',
               'AXP',
               'BA',
               'DD',
               'JNJ',
               'KO',
               'MCD',
               'MMM']
    #df= parseExcelFileWithMultipleSheetsAndCombine("/datadrive/Sahil/code/GL/fewTrails/twitter/Tweet-Scale.xlsx",SheetsToParse)
    df = pd.read_csv("/datadrive/Sahil/code/GL/fewTrails/twitter/twitter_training.csv")
    
    #df = pd.read_csv(training_data_csv, encoding='iso-8859-1')
    sentences_len = [len(str(s).split()) for s in df['text']]
    max_len = max(sentences_len) + 20 # 20 margin

    print( "Max Sentence length {}".format( max_len ) )

    V_index_dict = getIndexedDict(model)
    vocab_size = len(V_index_dict)
    embedding_weights = getEmbeddings(vocab_size, vec_len)

    data_X = []

    for sen in df.text[:]:
        #vec = np.zeros(max_len)
        vec = []
        for index, word in enumerate( word_tokenize( str( sen ) )[:max_len] ) :
            if word in V_index_dict.keys():
                vec.append(V_index_dict[word])
            else :
                vec.append(0)
        data_X.append(vec)

    data_X = np.array(data_X)

    data_X = sequence.pad_sequences(data_X, maxlen=max_len)

    y = df.Rating_m
    y = to_categorical(y, num_classes=None)
    print(y)
    print("Shape of Y{}".format(y.shape))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)

    for train_index, test_index1 in sss.split(data_X, y):
        print("TRAIN:", train_index, "TEST:", test_index1)
        print("TRAIN:", len(train_index), "TEST:", len(test_index1))
        X_train, X_test = data_X[train_index], data_X[test_index1]
        y_train, y_test = y[train_index], y[test_index1]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)

    for val_index, test_index2 in sss.split(X_test, y_test):
        print("TRAIN:", val_index, "TEST:", test_index2)
        print("TRAIN:", len(val_index), "TEST:", len(test_index2))
        X_val, X_test = X_test[val_index], X_test[test_index2]
        y_val, y_test = y_test[val_index], y_test[test_index2]

    data = {}
    
    data["X_train"] = X_train
    data["X_test"] = X_test
    data["X_val"] = X_val
    data["y_train"] = y_train
    data["y_test"] = y_test
    data["y_val"] = y_val

    data["train_index"] = train_index
    data["test_index"] = test_index1[test_index2]
    data["val_index"] = test_index1[val_index]

    data["max_len"] = max_len
    data["vec_len"] = vec_len
    data["vocab_size"] = vocab_size
    pickle.dump(data, open( saved_data_filename, 'wb'))

def load_data():
    print("Loading Data...")
    data = pickle.load( open( saved_data_filename, "rb" ) )

    X_train = data["X_train"]
    X_test = data["X_test"]
    X_val = data["X_val"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    y_val = data["y_val"]

    embedding_weights = np.load( os.path.join( wiki_data_path, embedding_path ) )

    max_len = data["max_len"]
    vec_len = data["vec_len"]
    vocab_size = data["vocab_size"]

    return X_train, X_test, X_val, y_train, y_test, y_val, embedding_weights, vec_len, vocab_size, max_len

def baseModel(lstm_size, dropout, activation):
    model = Sequential()
    print("LSTM_SIZE : {}, DROPOUT : {}, ACTIVATION : {}".format(lstm_size, dropout, activation))
    model.add(Embedding(vocab_size, vec_len, input_length = max_len, weights = [embedding_weights], trainable=False))
    #model.add( Bidirectional( LSTM (lstm_size, kernel_initializer="normal", dropout=dropout, activation=activation, name='lstm') ) )
    model.add( LSTM (lstm_size, kernel_initializer="normal", dropout=dropout, activation=activation, name='lstm1') )
    #model.add( LSTM (lstm_size/2, kernel_initializer="normal", dropout=dropout, activation=activation, name='lstm') )
    #model.add(LSTM(1024, kernel_initializer="normal"))
    #model.add(Flatten())
    model.add(Dense(256, kernel_initializer='uniform', activation='relu',name="dense1")) 
    model.add(Dense(128, kernel_initializer='uniform', activation='relu',name="dense2")) 
    model.add(Dense(output_dim=4, activation='softmax', name = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def get_accuracy(y_test, preds):
    preds_ = []
    y_ = []

    for p,y_t in zip(preds[:], y_test):
        preds_.append( np.argmax( p ) )
        y_.append( np.argmax( y_t ) )

    print("Accuracy on test data : ", accuracy_score(y_, preds_))

    cmat = confusion_matrix(y_, preds_)
    print("Class accuracies : ", cmat.diagonal()/cmat.sum(axis=1))

def train(model, lstm_size, drop, activation):
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=-0.01, patience=0, verbose=0, mode='auto')

    history = model.fit(X_train[:], y_train[:], epochs=epoch, 
                        #batch_size=batch, 
                        callbacks=[early_stopping], validation_data=(X_val[:], y_val[:]))
    print("Testing...")
    preds = model.predict(X_test[:], batch_size = batch)
    preds_val = model.predict(X_val[:], batch_size = batch)
    print("Val")
    get_accuracy(y_val, preds_val)
    print("Test")
    get_accuracy(y_test, preds)
    weights = model.get_weights()
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    name = str(lstm_size) + "_" + str(drop) + "_" + activation + "_b" + str(batch) + "_e" + str(epoch) 

    np.save(wts_name + name + "_softmax", layer_dict['softmax'].get_weights())
    np.save(wts_name + name + "_lstm", layer_dict['lstm1'].get_weights())
    np.save(wts_name + name + "_dense1", layer_dict['dense1'].get_weights())
    np.save(wts_name + name + "_dense2", layer_dict['dense2'].get_weights())

    K.clear_session()



#del X_train, X_val, X_test, embedding_weights

if __name__ == "__main__":
    nltk.data.path.append("/datadrive/common_datas/nltk")
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    wts_name = "wts6_"

    # Wiki and Embedding Saving paths
    wiki_model_name = "en.model"
    wiki_model_path = os.path.join("/datadrive","common_datas","word2vec_wiki","wikiModel")
    wiki_data_path = os.path.join("/datadrive","common_datas","word2vec_wiki")
    embedding_path = "wiki_embeddings.npy"
    indexed_vocab_path = "vocab_index.p"

    #Training Data paths
    training_data_csv = "final_v2.csv"
    saved_data_filename = "data_for_training_v5.p"

    print("on ", saved_data_filename)

    # Neural Configurations
    lstm_sizes = [512]
    #dropouts = [0,0.2,0.4, 0.5, 0.6, 0.8, 1]
    dropouts = [0.5]
    #activations = ['tanh', 'softsign']
    activations = ['tanh']


    # Get input parameters from online
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-e','--epoch', help='Number of epoch', required=False)
    parser.add_argument('-b','--batch', help='Batch Size', required=False)
    parser.add_argument('-n','--new', help='Generate new data?', required=False)
    parser.add_argument('-s','--seed', help='Seed for Data split', required=False)

    args = vars(parser.parse_args())

    create_data = True

    # Default epoch, seed, batch
    epoch = int(args['epoch']) if args['epoch'] != None else 10
    batch = int(args['batch']) if args['batch'] != None else 8
    seed = int(args['seed']) if args['seed'] != None else 123123

    if args['new'] != None:
        saved_data_filename = args['new']
    else :
        create_data = False

    print("Model will run for {} epoch with {} batch size".format(epoch, batch))
    print("Data from || ", training_data_csv)
    print("Data to || ", saved_data_filename)
    print("Split Seed : ", seed)

    #if saved_data_filname in os.listdir():
    if create_data == False:
        X_train, X_test, X_val, y_train, y_test, y_val, embedding_weights, vec_len, vocab_size, max_len = load_data()
    else :
        print("Generating Data...")
        prepareData()
        X_train, X_test, X_val, y_train, y_test, y_val, embedding_weights, vec_len, vocab_size, max_len = load_data()

    for size in lstm_sizes:
        for drop in dropouts:
            for activation in activations:
                model = baseModel(size, drop, activation)
                print(model.summary())
                train(model, size, drop, activation)
