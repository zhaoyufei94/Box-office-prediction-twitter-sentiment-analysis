#! /usr/bin/env python

# -*- coding:utf-8 -*-
# Author: Yufei Zhao

import os
import sys
import time
import nltk
import csv
import re
import string
import pickle
import random
from datetime import datetime
from collections import Counter
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def getData(path, sentiment, dataset):
	files = os.listdir(path)
	fileList = []
	for f in files:
		if(os.path.isfile(path + f)):
			if (f[0] == '.'):
				continue
			fileList.append(f)
	csvwriter = csv.writer(dataset, quotechar = '|')
	for f in fileList:
		fp = open(path + f, 'r')
		text = fp.readline()
		fp.close()
		csvwriter.writerow([sentiment, text.lower()])

# getFeatureVector: get feature vector of a row of text
# @row: the text from which we want feature vector (str)
#
# return: a list of stemmed words (utf-8) -> feature vector
#       may contain duplicated elements to indicate the 'importance' of this word
def getFeatureVector(row):
	regex = re.compile('[%s]' % re.escape(string.punctuation))
	snowball = SnowballStemmer('english')
	# tokenization
	try:
		token_text = word_tokenize(row.lower())
	except UnicodeDecodeError:
		return []
	cleaned_text = []
	for word in token_text:
		# remove punctuation
		word = regex.sub('', word)
		if word == '':
			continue
		# remove stop words
		if word in stopwords.words('english'):
			continue
		# stemming
		try:
			word = snowball.stem(word)
		except UnicodeDecodeError:
			continue
		# eliminate duplicated words
		if not word in cleaned_text:
			cleaned_text.append(word)
	featureVector = cleaned_text
	return featureVector

# getFeatureList: combine feature vectors to a global feature list (not in use)
def getFeatureList(features):
	featureList = []
	for line in features:
		for word in line[1]:
			#if not word in featureList:
			featureList.append(word)
	return featureList

# extractFeatures
# @review: a tuple of review text and sentiment
#
# return: trainable feature vector of review (same dimension with featureList)
def extractFeatures(review):
	review_words = set(review)
	features = {}
	for word in featureList:
		features['contains(%s)' % word] = (word in review_words)
	return features

def gettime():
    now = str(datetime.now()).split(".")[0].split()[1]
    return now

# Progress bar
# Adopted from Lustralisk's code
# http://www.cnblogs.com/lustralisk/p/pythonProgressBar.html
class ProgressBar(object):
    def __init__(self, count = 0, total = 0, width = 100):
        self.count = count
        self.total = total
        self.width = width
    def move(self):
        self.count += 1
    def log(self):
		percent = round(self.count * 100.0 / self.total, 1)
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        progress = self.width * self.count / self.total
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('>' * progress + '-' * (self.width - progress))
        sys.stdout.write('  [{0:3}%]'.format(percent) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()


if __name__ == "__main__":

	'''
 	dataset = open('./sentimentdata.csv','wb')
	getData('./Data/aclImdb/train/pos/', 'positive', dataset)
	getData('./Data/aclImdb/train/neg/', 'negative', dataset)
	dataset.close()
	'''

	print '[%s] starting import sentiment data' % gettime()
	dataset = open('./sentimentdata.csv', 'rb')
	csvreader = csv.reader(dataset, quotechar = '|')

	reviews = []
	featureList = []
	bar = ProgressBar(total = len(open('./sentimentdata.csv','rb').readlines()))

	# Load data from source file and generate features
	for row in csvreader:
		sentiment = row[0]
		featureVector = getFeatureVector(row[1])
		featureList.extend(featureVector)
		reviews.append((featureVector, sentiment))
		bar.move()
		bar.log()
	dataset.close()
	random.shuffle(reviews)
	print '[%s] successfully imported %d movie reviews' % (gettime(), len(reviews))

	# Filter top words in featureList to gain better-fitting model
	wordcounter = Counter(featureList)
	topwords = wordcounter.most_common(500)
	featureList = []
	for word in topwords:
		featureList.append(word[0])
	print '[%s] extracted top %d word features' % (gettime(), len(featureList))

	# Split training/testing data and generate trainable feature for each review
	tarin_size = len(reviews)*3/4
	train_set = nltk.classify.util.apply_features(extractFeatures, reviews[:tarin_size])
	test_set = nltk.classify.util.apply_features(extractFeatures, reviews[tarin_size:])
	print '[%s] start training on %d instances' % (gettime(), len(train_set))

	# Train and test Naive Bayes model
	NBClassifier = nltk.NaiveBayesClassifier.train(train_set)
	print '[%s] training completed' % gettime()
	print '[%s] start testing on %d instances' % (gettime(), len(test_set))
	print 'accuracy:', nltk.classify.util.accuracy(NBClassifier, test_set)
	NBClassifier.show_most_informative_features(5)

	# Store model
	#model = open('./NBClassifier.pickle', 'wb')
	#pickle.dump(NBClassifier, model)
	#model.close()
	#print NBClassifier.classify(extractFeatures(test_review))