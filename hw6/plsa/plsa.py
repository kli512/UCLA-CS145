from numpy import zeros, int8, log
from pylab import random
import sys
#import jieba
import nltk
from nltk.tokenize import word_tokenize 
import re
import time
import codecs

class PLSA(object):
    def initialize(self, N, K, M, word2id, id2word, X):
        self.word2id, self.id2word, self.X = word2id, id2word, X
        self.N, self.K, self.M = N, K, M
        # theta[i, j] : p(zj|di): 2-D matrix
        self.theta = random([N, K])
        # beta[i, j] : p(wj|zi): 2-D matrix
        self.beta = random([K, M])
        # p[i, j, k] : p(zk|di,wj): 3-D tensor
        self.p = zeros([N, M, K])
        for i in range(0, N):
            normalization = sum(self.theta[i, :])
            for j in range(0, K):
                self.theta[i, j] /= normalization;

        for i in range(0, K):
            normalization = sum(self.beta[i, :])
            for j in range(0, M):
                self.beta[i, j] /= normalization;


    def EStep(self):
        for i in range(0, self.N):
            for j in range(0, self.M):
                ## ================== YOUR CODE HERE ==========================
                ###  for each word in each document, calculate its
                ###  conditional probability belonging to each topic (update p)
                ps = self.beta[:, j] * self.theta[i, :]
                self.p[i, j] = ps / ps.sum()
                # ============================================================

    def MStep(self):
        # update beta
        for k in range(0, self.K):
            # ================== YOUR CODE HERE ==========================
            ###  Implement M step 1: given the conditional distribution
            ###  find the parameters that can maximize the expected likelihood (update beta)
            beta = (self.p[:, :, k] * self.X).sum(axis=0)
            self.beta[k] = beta / beta.sum()
            # ============================================================
        
        # update theta
        for i in range(0, self.N):
            # ================== YOUR CODE HERE ==========================
            ###  Implement M step 2: given the conditional distribution
            ###  find the parameters that can maximize the expected likelihood (update theta)
            theta = self.X[i] @ self.p[i]
            self.theta[i] = theta / theta.sum()
            # ============================================================


    # calculate the log likelihood
    def LogLikelihood(self):
        loglikelihood = 0
        for i in range(0, self.N):
            for j in range(0, self.M):
                # ================== YOUR CODE HERE ==========================
                ###  Calculate likelihood function
                loglikelihood += self.X[i, j] * log(self.theta[i] @ self.beta[:,j])
                # ============================================================
        return loglikelihood

    # output the params of model and top words of topics to files
    def output(self, docTopicDist, topicWordDist, dictionary, topicWords, topicWordsNum):
        # document-topic distribution
        file = codecs.open(docTopicDist,'w','utf-8')
        for i in range(0, self.N):
            tmp = ''
            for j in range(0, self.K):
                tmp += str(self.theta[i, j]) + ' '
            file.write(tmp + '\n')
        file.close()
        
        # topic-word distribution
        file = codecs.open(topicWordDist,'w','utf-8')
        for i in range(0, self.K):
            tmp = ''
            for j in range(0, self.M):
                tmp += str(self.beta[i, j]) + ' '
            file.write(tmp + '\n')
        file.close()
        
        # dictionary
        file = codecs.open(dictionary,'w','utf-8')
        for i in range(0, self.M):
            file.write(self.id2word[i] + '\n')
        file.close()
        
        # top words of each topic
        file = codecs.open(topicWords,'w','utf-8')
        for i in range(0, self.K):
            topicword = []
            ids = self.beta[i, :].argsort()
            for j in ids:
                topicword.insert(0, self.id2word[j])
            tmp = ''
            for word in topicword[0:min(topicWordsNum, len(topicword))]:
                tmp += word + ' '
            file.write(tmp + '\n')
        file.close()