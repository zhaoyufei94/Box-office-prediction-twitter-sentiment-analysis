#! /usr/bin/env python
# encoding=utf8

import os
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
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

regex = re.compile('[%s]' % re.escape(string.punctuation))
porter = PorterStemmer()
snowball = SnowballStemmer('english')


# Adopt Data into Labeled csv file
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

def getFeatureVector(row):
	# tokenization
	try:
		token_text = word_tokenize(row.lower())
	except UnicodeDecodeError:
		return []
	#except UnicodeDecodeError:
	#print 'token!', token_text
	cleaned_text = []
	for word in token_text:
		# remove punctuation
		#word = word.decode('utf8','ignore')
		#print 'original word: %s' % word
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
		#print 'processed word: %s' % word
		# eliminate duplicated words
		if not word in cleaned_text:
			cleaned_text.append(word)
	featureVector = cleaned_text
	return featureVector

def getFeatureList(features):
	featureList = []
	for line in features:
		for word in line[1]:
			#if not word in featureList:
			featureList.append(word)
	return featureList

def extractFeatures(review):
	review_words = set(review)
	features = {}
	for word in featureList:
		features['contains(%s)' % word] = (word in review_words)
	return features

def gettime():
    now = str(datetime.now()).split(".")[0].split()[1]
    return now


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
	for row in csvreader:
		sentiment = row[0]
		featureVector = getFeatureVector(row[1])
		featureList.extend(featureVector)
		reviews.append((featureVector, sentiment))
	dataset.close()
        random.shuffle(reviews)
        print '[%s] successfully imported %d movie reviews' % (gettime(), len(reviews))

	#result = open('./result.txt', 'wb')
	#model = open('./NBClassifier.pickle', 'wb')
    
        wordcounter = Counter(featureList)
        topwords = wordcounter.most_common(500)
        featureList = []
        for word in topwords:
            featureList.append(word[0])
        print '[%s] extracted top %d word features' % (gettime(), len(featureList))
	#featureList = list(set(featureList))
	tarin_size = len(reviews)*3/4
	train_set = nltk.classify.util.apply_features(extractFeatures, reviews[:tarin_size])
	test_set = nltk.classify.util.apply_features(extractFeatures, reviews[tarin_size:])
	print '[%s] start training on %d instances' % (gettime(), len(train_set))

	NBClassifier = nltk.NaiveBayesClassifier.train(train_set)
	print '[%s] training completed' % gettime()
	#pickle.dump(NBClassifier, model)
	#model.close()
	#print NBClassifier.classify(extractFeatures(test_review))
	print '[%s] start testing on %d instances' % (gettime(), len(test_set))
	print 'accuracy:', nltk.classify.util.accuracy(NBClassifier, test_set)
	NBClassifier.show_most_informative_features(5)



