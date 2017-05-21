# Sentiment Classifier
# LING 406 Intro to Computational Linguistics
# University of Illinois at Urbana-Champaign
# Cheng-Yang Peter Liu (cyliu4), 
# May 8th, 2017
#
# This file includes three functions.
# 1. add_feature: add selected feature to the document
# 2. bow: create Bag-of-Words feature from documents
# 3. find_feature: a part of bow, create word list 

import nltk
import re
from nltk.sentiment import SentimentAnalyzer
from nltk.corpus import stopwords

def find_features(document, word_features):
	# create word list
	words = set(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features

def bow(documents, word_size):
	# create bow feature
	all_words = []
	for files in documents:
		for w in files[0]:
			all_words.append(w.lower()) # put low words into a list

	all_words = nltk.FreqDist(all_words) # find the frequency of words
	word_features = list(all_words.keys())[:word_size] # use only top frequency words
	# create feature
	featuresets = [(find_features(rev, word_features), category) for (rev, category) in documents]

	return featuresets

def add_feature(documents, remove_stopwords, include_neg, keep_adjv, stemming):
	# remove stopwrods
	if remove_stopwords:
		i = 0
		for files, cats in documents:
			for sw in stopwords.words('english'): # use pre-compiled list
				files = [w for w in files if w!=sw] # remove them 
			documents[i] = tuple([files,cats])
			i += 1

	# include negation
	if include_neg:
		sentim_analyzer = SentimentAnalyzer() # use nltk library
		documents = [nltk.sentiment.util.mark_negation(doc) for doc in documents]

	# include only adjective and adverb
	if keep_adjv:
		i = 0
		for files, cats in documents:
			tagged = nltk.pos_tag(files) # create part of speech tag
			files = [w for w, tag, in zip(files,tagged) 
				 if re.search(r'^(JJ|RB)R?S?$',  tag[1])] # find adj, adv
			documents[i] = tuple([files,cats]) # create new document
			i += 1

	# use stemming
	if stemming:
		ps = nltk.stem.PorterStemmer() # use Porter Stemmer
		i = 0
		for files, cats in documents:
			temp = []
			for w in files:
				if w == 'oed': # nltk bug: http://stackoverflow.com/questions/41517595/nltk-stemmer-string-index-out-of-range
					continue
				temp.append(ps.stem(w)) # create new list
			documents[i] = tuple([temp, cats]) # create new document
			i += 1
		# documents = [([ps.stem(w) for w in files], cats) for files, cats in documents ]

	return documents