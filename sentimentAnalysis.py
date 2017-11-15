#! /usr/bin/env python
# encoding=utf8

import os
import nltk
import csv
import re
import string
import pickle
from datetime import datetime
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
			if not word in featureList:
				featureList.append(word)
	return featureList

def extractFeatures(review):
	review_words = set(review)
	features = {}
	for word in featureList:
		features['contains(%s)' % word] = (word in review_words)
	return features

if __name__ == "__main__":
	'''
 	dataset = open('./sentimentdata.csv','wb')
	getData('./Data/aclImdb/train/pos/', 'positive', dataset)
	getData('./Data/aclImdb/train/neg/', 'negative', dataset)
	dataset.close()
	'''
	dataset = open('./sentimentdata.csv', 'rb')
	csvreader = csv.reader(dataset, quotechar = '|')

	reviews = []
	featureList = []
	for row in csvreader:
		sentiment = row[0]
		featureVector = getFeatureVector(row[1])
		#print row[1]
		#print featureVector
		featureList.extend(featureVector)
		reviews.append((featureVector, sentiment))
	#print reviews
	#print reviews
	dataset.close()

	#result = open('./result.txt', 'wb')
	model = open('./NBClassifier.pickle', 'wb')

	featureList = list(set(featureList))
	tarin_size = len(reviews)*3/4
	train_set = nltk.classify.util.apply_features(extractFeatures, reviews[:tarin_size])
	test_set = nltk.classify.util.apply_features(extractFeatures, reviews[tarin_size:])
	#result.write('')
	now = str(datetime.now()).split(".")[0].split()[1]
	print '[',now,'] train on %d instances, test on %d instances' % (len(train_set), len(test_set))

	NBClassifier = nltk.NaiveBayesClassifier.train(train_set)
	now = str(datetime.now()).split(".")[0].split()[1]
	print '[',now,'] training complete, start to store classifier model'
	pickle.dump(NBClassifier, model)
	model.close()
	#test_review = 'Bromwell High is a cartoon comedy. It ran at the same time as some other programs about school life, such as "Teachers". My 35 years in the teaching profession lead me to believe that Bromwell Highs satire is much closer to reality than is "Teachers". The scramble to survive financially, the insightful students who can see right through their pathetic teachers pomp, the pettiness of the whole situation, all remind me of the schools I knew and their students. When I saw the episode in which a student repeatedly tried to burn down the school, I immediately recalled ......... at .......... High. A classic line: INSPECTOR: Im here to sack one of your teachers. STUDENT: Welcome to Bromwell High. I expect that many adults of my age think that Bromwell High is far fetched. What a pity that it isnt!'
	#test_review = 'Once again Mr. Costner has dragged out a movie for far longer than necessary. Aside from the terrific sea rescue sequences, of which there are very few I just did not care about any of the characters. Most of us have ghosts in the closet, and Costners character are realized early on, and then forgotten until much later, by which time I did not care. The character we should really care about is a very cocky, overconfident Ashton Kutcher. The problem is he comes off as kid who thinks hes better than anyone else around him and shows no signs of a cluttered closet. His only obstacle appears to be winning over Costner. Finally when we are well past the half way point of this stinker, Costner tells us all about Kutchers ghosts. We are told why Kutcher is driven to be the best with no prior inkling or foreshadowing. No magic here, it was all I could do to keep from turning it off an hour in.'
	#test_review = 'The only good thing about this movie was the shot of Goldie Hawn standing in her little french cut bikini panties and struggling to keep a dozen other depraved women from removing her skimpy little cotton top while she giggled and cooed. Ooooof! Her loins rival those of Nina Hartley. This movie came out when I was fourteen and that shot nearly killed me. Id forgotten about it all tucked away in the naughty Roladex of my mind until seeing it the other day on TV, where they actually blurred her midsection in that scene, good grief, reminding me what a smokin hottie of a woman Goldie Hawn was in the 80s. Kurt Russell must have had a fun life.'
	#test_review = 'bad sad unfortunately ridiculous dislike hate awkward'
	#print NBClassifier.classify(extractFeatures(test_review))
	#NBClassifier = nltk.NaiveBayesClassifier.train(train_set)
	now = str(datetime.now()).split(".")[0].split()[1]
	print '[',now,'] storing complete, start to test classifier model'
	print 'accuracy:', nltk.classify.util.accuracy(NBClassifier, test_set)
	NBClassifier.show_most_informative_features()



