#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""generate_data.py: Re-generate the word2vec file in order to reduce its size."""
__author__ = "Anthony FARAUT"

def read_word_from_word2vec_model(filename, corpus_set):
    """ Read words from word2vec model
    Parameters
    ----------
    filename : The filename of the corpus file
    corpus_set :  A set containing unique words from the corpus

    """
    fin = open(filename,'r')
    fout = open(".".join(filename.split('.')[:-1])+'.edited.'+filename.split('.')[-1], 'w')
    for line in fin:
        splited_line = line.strip().split()
        if splited_line[0] in corpus_set:
            fout.write(line)


def read_word_from_corpus(filename):
    """ Read words from corpus
    Parameters
    ----------
    filename : The filename of the corpus file

    Returns
    -------
    set : A set of unique words from the corpus
    """
    with open(filename) as fin:
        raw_text = fin.read()
        raw_text = raw_text.replace('\n',' ').replace(',', ' ').replace('"', '').replace("*", "").replace("...", " ").replace(".", " ").replace("!", " ").replace("?", " ")
    return set(raw_text.split(" "))

def main():
    corpus = read_word_from_corpus('../data/bigshake.txt')
    read_word_from_word2vec_model("../data/glove.6B.50d.txt", corpus)


if __name__ == '__main__':
    main()