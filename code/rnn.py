#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nltk, theano, math
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
from codecs import open
from keras.layers import Dense, Activation, Embedding
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from math import floor
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import Callback

"""ner.py: Building a Language Model using Recurrent Neural Networks."""
__author__ = "Anthony FARAUT"

BIGTXT = "../data/bigshake.txt"
LSTM_OUTPUT_DIM = 500
VERBOSE = 2
NB_EPOCH = 10

"""
Class Perplexity for the 'on epoch end' event
"""
class Perplexity(Callback):

    def __init__(self, x_test):
        self.x_test = x_test

    def on_epoch_end(self, epoch, logs={}):
        """ Callback on epoch end - Excecuted each time on the epoch end event
        Parameters
        ----------
        self : Class elements
        epoch : The current epoch
        logs : The logs

        """
        y_test_predicted = self.model.predict(self.x_test)
        result, count = 0, 0
        for index_sentence, sentence in enumerate(y_test_predicted.tolist()):
            for index_word, word in enumerate(sentence):
                result+= math.log(max(word), 2)
                count+=1

        result = 2 ** ((float(-1)/count) * result)
        print("# ---------------------")
        print("Perplexity on epoch " + str(epoch + 1) + " -> " + str(result))
        print("# --------------------- End of the epoch")


def read_wordvecs(filename):
    """ Read word vectors from word2vec file
    Parameters
    ----------
    filename : The filename of the word2vec file

    Returns
    -------
    array : An array containing the values for each word2vec word
    defaultdict : word -> index
    defaultdict : index -> word 
    """
    fin = open(filename, encoding='utf8')
    
    word2index = defaultdict(lambda:1)
    index2word = dict()
    
    # Masking
    word2index['MASK'] = 0
    # Out of vocabulary words
    word2index['UNK'] = 1
    # End-Of-Sentence
    word2index['EOS'] = 2
    
    index2word[0] = 'MASK'
    index2word[1] = 'UNK'
    index2word[2] = 'EOS'

    word_vecs = []
    
    for line in fin:
        splited_line = line.strip().split()
        word = splited_line[0]
        word_vecs.append(splited_line[1:])
        
        word2index[word] = len(word2index)
        index2word[len(index2word)] = word
    
    word_vecs_np = np.zeros(shape=(len(word2index), len(word_vecs[1])), dtype='float32')
    word_vecs_np[3:] = word_vecs
    
    return word_vecs_np, word2index, index2word


def get_list_of_tokenized_sentences_nltk(): 
    """ Get list of tokenized sentences with nltk method

    Returns
    -------
    array : The sentences tokenierd
    """
    sentences = []
    with open(BIGTXT) as fin:
        raw_text = fin.read()

    raw_text = raw_text.replace('\n', ' ').replace(',', ' ').replace('"', '').replace("*", "").replace("...", " EOS ").replace(".", " EOS ").replace("!", " EOS ").replace("?", " EOS ")
    list_sentences = raw_text.split("EOS")
    list_sentences = [ sent + " EOS" for sent in list_sentences]
    return [nltk.word_tokenize(sent) for sent in list_sentences]


def get_indexes_from_tokenized_sentences(tokenized_sentences, word2index):
    """ Get the index of each tokenized sentence
    Parameters
    ----------
    tokenized_sentences : The array of the tokenized sentences
    word2index : The dict containing the indexes

    Returns
    -------
    array : The indexes of each tokenized sentence
    """
    indexes = []
    for sentence in tokenized_sentences:
        index = []
        for token in sentence:
            if not token == "EOS":
                token = token.lower()
            index.append(word2index[token])
        indexes.append(index)
    return indexes


def create_rnn_model(word2index, word_vecs):
    """ Create the Rnn model with Keras
    Parameters
    ----------
    word2index : A dict which match word -> index
    word_vecs : The word2vec vectors for each word

    Returns
    -------
    Keras model : The keras model created
    """
    model = Sequential()

    el = Embedding(input_dim=len(word_vecs), output_dim=50, weights=[word_vecs])
    model.add(el)

    lstm = LSTM(LSTM_OUTPUT_DIM, return_sequences=True)
    model.add(lstm)

    dense = Dense(len(word2index), activation='softmax')
    timeDistributed = TimeDistributed(dense)
    model.add(timeDistributed)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def build_x_y_dataset(train, test):
    """ Build all datasets 
    Parameters
    ----------
    train : The train dataset
    test : The test dataset

    Returns
    -------
    array : The X train dataset
    array : The Y train dataset
    array : The X test dataset
    array : The Y test dataset
    """
    x_train, y_train, x_test, y_test = [], [], [], []
    for index_train in train: # -- Train
        x_train.append(index_train[:-1])
        y_train.append(index_train[1:])

    for index_test in test: # -- Test
        x_test.append(index_test[:-1])
        y_test.append(index_test[1:])

    return x_train, y_train, x_test, y_test


def pad_and_share_all_corpus(x_train, y_train, x_test, y_test):
    """ Pad and share all the datasets
    Parameters
    ----------
    x_train : The X train dataset
    y_train : The Y train dataset
    x_test : The X test dataset
    y_test : The Y test dataset

    Returns
    -------
    array : The X train dataset padded
    array : The Y train dataset padded
    array : The X test dataset padded
    array : The Y test dataset padded
    """
    x_train = pad_sequences(x_train)
    y_train = pad_sequences(y_train)
    x_test = pad_sequences(x_test)
    y_test = pad_sequences(y_test)

    x_train_shared = theano.shared(value=x_train, name='x_train')
    y_train_shared = theano.shared(value=y_train, name='y_train')
    x_test_shared = theano.shared(value=x_test, name='x_test')
    y_test_shared = theano.shared(value=y_test, name='y_test')

    return x_train, y_train, x_test, y_test



def train_model(model, x_train, y_train, x_test, y_test):
    """ Train the model
    Parameters
    ----------
    model : The rnn model
    x_train : The X train dataset
    y_train : The Y train dataset
    x_test : The X test dataset
    y_test : The Y test dataset

    """
    model.fit(x_train, y_train, nb_epoch=NB_EPOCH, verbose=VERBOSE, callbacks=[Perplexity(x_test)])

    print 'Testing...'
    res = model.evaluate(x_test, y_test, verbose=VERBOSE)

    print('Test accuracy loss: ', res)

    loss = model.evaluate(x_train, y_train, verbose=VERBOSE)
    print('Train accuracy loss: ', loss)


def keras_predict(model, x_test, y_test):
    """ Predict the model
    Parameters
    ----------
    model : The rnn model
    x_test : The X test dataset
    y_test : The Y test dataset

    """
    print '# --- predict'

    output_array = model.predict(x_test)
    len_output_array = len(output_array)

    tp, numberOfWord = 0, 0
    for index_sentence, sentence in enumerate(output_array.tolist()):
        for index_word, word in enumerate(sentence):
            numberOfWord+=1
            tp += y_test[index_sentence][index_word][word.index(max(word))]

    result = str("{} / {}".format(tp, numberOfWord))
    percentage = float(float(tp)/int(numberOfWord)) * 100

    print("result ", result)
    print("percentage ", percentage)


def create_vector_one_hot_from_word(dictkeys, length, word):
    """ Create one hot vector from word
    Parameters
    ----------
    dictkeys : The words index
    length : The length of each vector
    word : The word we want the one hot vector

    Returns
    -------
    numpy array : The one hot encoded vector for the word
    """
    vec = np.zeros(length, dtype=np.float32)
    index = dictkeys.index(word)
    vec[index-1] = 1
    return np.asarray(vec, dtype=np.float32)


def one_hot_y (y_corpus, word2index, index2word):
    """ Generate each one hot vector
    Parameters
    ----------
    y_corpus : The Y test dataset
    word2index : word -> index
    index2word : index -> word

    Returns
    -------
    array : The Y test dataset modified in one hot encoded vectors
    """
    dictkeys = word2index.keys()
    dictkeys.sort()
    length = len(dictkeys)

    y_modified = []
    for y_sentences in y_corpus:
        tmp = []
        for y in y_sentences:
            tmp.append(create_vector_one_hot_from_word(dictkeys, length, index2word[y]))
        y_modified.append(tmp)
    return y_modified


def main():

    word_vecs_np, word2index, index2word = read_wordvecs("../data/glove.6B.50d.edited.txt")

    tokenized_sentences = get_list_of_tokenized_sentences_nltk()

    indexes = get_indexes_from_tokenized_sentences(tokenized_sentences, word2index)

    # Train 80 % - Test 20 %
    eighty_percent = int(floor(0.8 * len(indexes)))
    train = indexes[:eighty_percent]
    test = indexes[eighty_percent + 1:]

    x_train, y_train, x_test, y_test = build_x_y_dataset(train, test)
    x_train, y_train, x_test, y_test = pad_and_share_all_corpus(x_train, y_train, x_test, y_test)
    
    model = create_rnn_model(word2index, word_vecs_np)

    y_train_modified = one_hot_y(y_train, word2index, index2word)
    y_test_modified = one_hot_y(y_test, word2index, index2word)

    train_model(model, x_train, y_train_modified, x_test, y_test_modified)
    keras_predict(model, x_test, y_test_modified)


if __name__ == '__main__':
    main()