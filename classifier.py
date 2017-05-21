# Sentiment Classifier
# LING 406 Intro to Computational Linguistics
# University of Illinois at Urbana-Champaign
# Cheng-Yang Peter Liu (cyliu4), 
# May 8th, 2017
#
# This file includes four machine learning functions and one cross validation function.

import numpy as np
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def cv(featuresets, classifier):
	# Cross validation
	fold_num = 5
	file_size = len(featuresets)
	subset_size = file_size/fold_num
	accuracy = []

	for fold in xrange(fold_num):
		training_set = featuresets[:fold*subset_size] + featuresets[(fold+1)*subset_size:] # choose training set
		testing_set = featuresets[fold*subset_size:][:subset_size] # choose testing set
		accuracy.append((nltk.classify.accuracy(classifier.train(training_set), testing_set))*100) # do training and testing
	return np.mean(accuracy)

def nb(featuresets):
	# Naive Bayes classifier
	classifier = nltk.NaiveBayesClassifier
	accuracy = cv(featuresets, classifier)
	print "Naive Bayes Classifier accuracy percent: ", accuracy

	return accuracy

def lg(featuresets):
	# Logistic regression (aka logit, MaxEnt) classifier
	classifier = SklearnClassifier(LogisticRegression())
	accuracy = cv(featuresets, classifier)
	print "Logistic Regression Classifier accuracy percent: ", accuracy

	return accuracy

def dt(featuresets):
	# Decision tress classifier
	classifier = SklearnClassifier(DecisionTreeClassifier())
	accuracy = cv(featuresets, classifier)
	print "Decision Tree Classifier accuracy percent: ", accuracy

	return accuracy

def lsvm(featuresets):
	# Support vector machine (SVM) classifier
	classifier = SklearnClassifier(LinearSVC())
	accuracy = cv(featuresets, classifier)
	print "Linear SVM classifier accuracy percent: ", accuracy

	return accuracy
