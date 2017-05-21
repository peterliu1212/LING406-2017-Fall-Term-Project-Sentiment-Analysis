# Sentiment Classifier
# LING 406 Intro to Computational Linguistics
# University of Illinois at Urbana-Champaign
# Cheng-Yang Peter Liu (cyliu4), 
# May 8th, 2017
#
# This is a the main file to do the experiment.

import nltk
import random

import classifier
import feature
import read_data

def run_exp(documents, save_file, tune_para):
	file = open(save_file, 'w')
	acy = []
	for parameter in tune_para:
		print 'parameter: ', parameter
		featuresets = feature.bow(documents, parameter) # generate Bag-of-Words feature
		
		# Classification
		print 'Start Naive Bayes Classification'
		nb_result = classifier.nb(featuresets) # get the accuracy from Naive Bayes 
		print 'Start Logistic Regression Classification'
		lg_result = classifier.lg(featuresets) # get the accuracy from logistic regression 
		print 'Start Decision Tree Classification' 
		dt_result = classifier.dt(featuresets) # get the accuracy from decision tree
		print 'Start SVM Classification'
		lsvm_result = classifier.lsvm(featuresets) # get the accuracy from support vector machine

		# save the result to accuracy file
		acy.append([nb_result, lg_result, dt_result, lsvm_result])
		file.write("%s, %s, %s, %s, %s\n" % (parameter, nb_result, lg_result, dt_result, lsvm_result))

	file.close()

# Choose dataset
# dataset = 0 # movie review
dataset = 1 # yelp review
# Read data and select features
feat = [False, False, False, False] # feat = [remove_stopwords, include_neg, keep_adjv, stemming]
if dataset == 0:
	documents = read_data.movie() # read movie data
	save_dataset = 'movie' 
else:
	documents = read_data.yelp() # read yelp data
	save_dataset = 'yelp'
documents = feature.add_feature(documents, feat[0], feat[1], feat[2], feat[3]) # add feature
random.shuffle(documents) # shuffle the data for advoid bias

# Run experiment
print 'Experiment 1: Baseline, Bag of Word' # print out process
# Generate feature sets
para = range(100,1000,200) + range(1000,10000,2000) + range(10000, 40000, 10000) # parameter size
# Feature selection
# save result file name
save_feature = 'bow'
save_para = '100_30000'
save_file = save_dataset + '_' + save_feature + '_' + save_para + '.txt'
# run experiment
para_bow = run_exp(documents, save_file, para)

