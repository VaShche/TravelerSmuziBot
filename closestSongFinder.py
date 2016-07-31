# -*- coding: utf-8 -*-

import os

dataDir="TSBsongs"
w2vDicFile="cbow-py3.dic"
#w2vMdlFile="cbow_ns300_fullrostelLK4.npy"
w2vMdlFile="cbow-py3.npy"

##########################################################
#w2v by Smirnov
###########################################################
import sys, getopt
import copy
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import NotFittedError
import re
import struct
import logging

class Word2Vec_mikolovmodel:
    """ Provide functionality to load and use word2vec models """

    def __init__(self):
        self.w2v = 0  # word2vec model
        self.dict = {}  # word2vec dictionary
        self.num_of_occur = []  # number of occurrences for words in the dictionary
        self.dim = 0  # dimensionality of words vector space
        self.num_words = 0  # the total number of words

    def load_word2vec_model(self, w2v_file):
        """
            loads word2vec model
            w2v_file -- file with word2vec model
            w2v_file: format: first 8 bytes = 2 uints for the number of words and dimensionality of w2v space
                         the rest of the file -- w2v matrix [dim, num_words]
        """
        npdata = np.fromfile(w2v_file, 'f')  # read the data from w2v_file
        self.num_words = struct.unpack('I', struct.pack('f', npdata[0]))[0]
        self.dim = struct.unpack('I', struct.pack('f', npdata[1]))[0]
        self.w2v = np.reshape(npdata[2:], (self.dim, self.num_words), 'F')


    def load_word2vec_dictionary(self, dict_file):
        """
            loads word2vec dictionary
            dict_file -- file with word2vec dictionary
            dict_file: format: word number_of_occurrences
        """
        with open(dict_file) as dictionary:
            line_num = 0
            if self.num_of_occur:
                self.num_of_occur = []
                logging.info("Nonempty vocabulary in Word2Vec Class. That's weird.")
            for line in dictionary:
                line = re.sub('\s+$', '', line)
                cur_word, cur_num_of_occur = re.split('\s+', line)
                self.dict[cur_word] = line_num
                self.num_of_occur.append(float(cur_num_of_occur))
                line_num += 1

    def covert_from_words_to_vecs(self, word_data):
        """
            convert words from word_data to vectors representations
            :param word_data: list of lists of words
            :return: list of np arrays of vector representations ((dim v2v) x (number of words))
        """
        num_sent = len(word_data)
        wordvec_data = [None] * num_sent
        for sent_num in range(num_sent):
            # print sent_num
            num_words = len(word_data[sent_num])
            wordvec_data[sent_num] = np.zeros((self.dim, num_words))
            for word_num in range(num_words):
                try:
                    cur_word_position = self.dict[word_data[sent_num][word_num]]
                    wordvec_data[sent_num][:, word_num] = self.w2v[:, cur_word_position]
                except KeyError:
                    curWord = word_data[sent_num][word_num]
                    logging.debug("Can't find the word "+ curWord +" in dictionary. ")
        return wordvec_data

    def test_word2vec(self):
        """ for debugging """
        print ('Test w2v dictionary')
        c = 0
        for wd in self.dict.keys():
            c += 1
            if c < 10:
                print (wd, self.dict[wd])
        print ('Test w2v matrix')
        print (self.w2v[0:3, 0:3])
        # result for cbow_ns
        # 0,00133422855287800	-0,145330294966698	0,0611438602209091
        # 0,00147313438355923	0,321542859077454	0,581905424594879
        # -0,00127675372641500	-0,155739039182663	-0,226470172405243

    @staticmethod
    def test_convert_from_words_to_vecs(word_data, vec_data):
        """ for debugging """
        outlength_worddata = 1
        outlength_vecs = 5
        rp_worddata = np.random.permutation(len(word_data))
        rp_vecs = np.sort(np.random.permutation(len(vec_data[0][:, 0]))[0:outlength_vecs])
        for sn in range(outlength_worddata):
            for wn in range(len(word_data[rp_worddata[sn]])):
                print (word_data[rp_worddata[sn]][wn])
                print (rp_vecs)
                print (vec_data[rp_worddata[sn]][rp_vecs, wn])


class Word2Vec(Word2Vec_mikolovmodel):
    def __init__(self):
        Word2Vec_mikolovmodel.__init__(self)

    def load_word2vec_model(self, w2v_file):
        """
            loads word2vec model
            w2v_file -- file with word2vec model
            w2v_file: format: first 8 bytes = 2 uints for the number of words and dimensionality of w2v space
                         the rest of the file -- w2v matrix [dim, num_words]
        """
        self.w2v = np.load(w2v_file)
        #np.savetxt("_w2v.csv",self.w2v)
        #print("qq")
        self.num_words = np.shape(self.w2v)[1]-1
        self.dim = np.shape(self.w2v)[0]

    def covert_from_words_to_vecs(self, word_data):
        """
            convert words from word_data to vectors representations
            :param word_data: list of lists of words
            :return: list of np arrays of vector representations ((dim w2v) x (number of words))
        """
        num_sent = len(word_data)
        wordvec_data = [None] * num_sent
        for sent_num in range(num_sent):
            num_words = len(word_data[sent_num])
            # in the last column of w2v there is a vector for unknown words
            wordvec_data[sent_num] = np.zeros((self.dim, num_words)) + np.reshape(self.w2v[:, -1], (self.dim, 1))
            for word_num in range(num_words):
                try:
                    cur_word_position = self.dict[word_data[sent_num][word_num]]
                    wordvec_data[sent_num][:, word_num] = self.w2v[:, cur_word_position]
                except KeyError:
                    curWord = word_data[sent_num][word_num]
                    logging.debug("Can't find the word "+ curWord +" in dictionary. ")
        return wordvec_data


class Preprocessor:
    """ to get info for preprocessing from word2vec model and to preprocess words vectors"""
    def __init__(self, preproc_type='whitening'):
        self.vocabulary = []
        self.allowed_preproc_types = ['whitening', ]
        self.preproc_type = preproc_type
        self.Mean = []
        self.Cov = []
        self.SqrtCov = []

    def get_mean_and_covariance(self, w2v, num_of_occurences):
        """ get mean and covariance of words vectors over the training set of word2vec model
            w2v -- word2vec model (in matrix form)
            num_of_occurences -- array that specifies weights for averaging over words
        """
        weights = num_of_occurences/np.sum(num_of_occurences)
        try:
            w2v_temp = np.multiply(w2v, weights)
        except MemoryError:
            w2v_temp = np.copy(w2v)
            for wn in range(np.shape(w2v)[1]):
                w2v_temp[:, wn] *= weights[wn]
        self.Mean = np.sum(w2v_temp, 1)
        try:
            w2v_except0 = w2v - np.reshape(self.Mean, (len(self.Mean), 1))
        except MemoryError:
            w2v_except0 = w2v_temp  # just to set the right shape (to avoid memoryError)
            for wn in range(np.shape(w2v)[1]):
                w2v_except0[:, wn] = w2v[:, wn] - self.Mean

        try:
            w2v_normalized = np.multiply(w2v_except0, np.power(weights, 0.5))
        except MemoryError:
            w2v_normalized = w2v_except0
            for wn in range(np.shape(w2v_except0)[1]):
                w2v_normalized[:, wn] *= weights[wn]**0.5

        self.Cov = np.dot(w2v_normalized, np.transpose(w2v_normalized))
        self.Cov = self.Cov/np.shape(w2v)[1]
        self.SqrtCov = fractional_matrix_power(self.Cov, -0.5)

    def preproc_wordvecs(self, wordvecs):
        """ preprocess words vectors.
            wordvecs -- initial words vectors
            return wordvecs_proc -- preprocessed wordvectors
        """
        wordvecs_proc = copy.deepcopy(wordvecs)
        if self.preproc_type not in self.allowed_preproc_types:
            print ("Unknown preprocessing type. Using whitening instead...")
        if not (np.any(self.SqrtCov) and np.any(self.Mean)):
            print ("Mean or covariance hasn't been set yet. I am leaving data unpreprocessed... ")
        else:
            num_sent = len(wordvecs)
            for sent_num in range(num_sent):
                wordvecs_proc[sent_num] = np.dot(self.SqrtCov, wordvecs_proc[sent_num] - np.reshape(self.Mean, (len(self.Mean), 1)))
        return wordvecs_proc

    def test_get_mean_and_covariance(self):
        """ for debugging """
        dim = 5
        print ("Mean:")
        print (self.Mean[0:dim])
        print ("Cov:")
        print (self.Cov[0:dim, 0:dim])
        print ("SqrtCov:")
        print (self.SqrtCov[0:dim, 0:dim])


class Reducer:
    """ to obtain sentence vector from words vectors """
    def __init__(self):
        self.allowed_reduction_types = ['average', ]
        self.reduction_type = 'average'

    def wordvec2sentvec(self, wordvecs):
        """ processes words vectors to obtain sentence vectors
            wordvecs -- words vectors (list of arrays (dim_of_vector_space x number_of_words_in_sentence))
            return sentvecs -- sentences vectors (array (dim_of_vector_space x number_of_sentences))
        """
        if self.reduction_type not in self.allowed_reduction_types:
            logging.warning("Unknown reduction type. Using 'average' instead")
            self.reduction_type = 'average'
        num_sent = len(wordvecs)
        dim_wordvecs = len(wordvecs[0][:, 0])
        sentvecs = np.zeros((dim_wordvecs, num_sent))
        for sent_num in range(num_sent):
            sentvecs[:, sent_num] = np.mean(wordvecs[sent_num], axis=1)
        return sentvecs

class Word2VecWrap():
    def __init__(self, path_to_w2v_model, path_to_w2v_dict):
        # load w2v model
        self.model_path = path_to_w2v_model
        self.dict_path = path_to_w2v_dict
        self.word2vec = Word2Vec()  # contains w2v model and dictionary, can convert words to wordvecs
        logging.info("Loading w2v model...")
        self.word2vec.load_word2vec_model(self.model_path)
        logging.info("Loading w2v dictionary...")
        self.word2vec.load_word2vec_dictionary(self.dict_path)

        # initialize reducer (can make sentence vectors from word vectors)
        self.reducer = Reducer()


##############################################################
# main
##############################################################




print ("load w2v")
w2v = Word2VecWrap(w2vMdlFile, w2vDicFile)
print ("done")

import re
reSpace=re.compile('[\r\n\.\-\,\—]')
def processSong(line):
    return reSpace.sub(" ",line).lower().replace("ё","е").replace("\u0301",'')
    #return line.replace('\n',' ').replace('\r',' ').replace(","," ").replace("."," ").replace("-"," ").replace("ё","е").replace("Ё","Е").lower()
pass

def song2vec(songText):
    words=songText.split(' ')
    wordsLst=[words]
    wordvecs=w2v.word2vec.covert_from_words_to_vecs(wordsLst)
    print (len(wordvecs),wordvecs[0].shape)
    sentVec=w2v.reducer.wordvec2sentvec(wordvecs)
    #print(sentVec.shape)
    return sentVec
pass

id2Song={}
id2ProcSong={}
id2vec={}
for fname in os.listdir(dataDir):
    print("**** "+fname)
    with open(os.path.join(dataDir, fname), "rt") as f:
        #lines=f.readlines()
        line=f.read()

        id2Song[fname]=line
        processedSong=processSong(line)
        id2ProcSong[fname]=processedSong

        id2vec[fname]=song2vec(processedSong)

#        print (">"+line)
    pass
pass

import scipy
import scipy.spatial

def getClosestSongIdToString(qtxt):
    qptxt=processSong(qtxt)
    print(qptxt)
    qvec=song2vec(qptxt)
    smin=3
    sidmin=-1
    for sid,svec in id2vec.items():
        #ed=np.linalg.norm(qvec-svec)
        dist=scipy.spatial.distance.cdist(np.atleast_2d(qvec).T, np.atleast_2d(svec).T, 'cosine')
        if dist<smin:
            smin=dist
            sidmin=sid
        #print (sid,dist)
    pass
    print(sidmin)
    return (sidmin)
pass

def getSongTxtById(sid):
    with open(os.path.join(dataDir, sid) ,"rt") as f:
        line=f.read()
        return line
pass

if __name__ == "__main__":
    #qtxt="любовь и голуби поцелуи"
    qtxt="дворцовая набережная Невы в центре Санкт-Петербурга находится по левому берегу от Набережной Кутузова до Адмиралтейской набережной. На набережной расположены здания Государственного Эрмитажа, Русского музея и пр."
    #qtxt="литейный мост — разводной мост через Неву в Санкт-Петербурге. Соединяет центральную часть города по оси Литейного проспекта с Выборгской стороной улица Академика Лебедева. Второй постоянный мост через Неву после Благовещенского моста."
    #qtxt="петропавловская крепость крепость в Санкт-Петербурге, расположенная на Заячьем острове, историческое ядро города. Официальное название — Санкт-Петербургская, в 1914—1917 годах — Петроградская крепость ."
    #qtxt="Соборная мечеть Санкт-Петербурга  памятник архитектуры, стиль северный модерн, главная мечеть Российской империи, крупнейшая мечеть в европейской части Российской империи"

    print (getClosestSongIdToString(qtxt))