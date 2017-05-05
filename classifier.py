import numpy as np
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def cv(featuresets, classifier):
	fold_num = 5
	file_size = len(featuresets)
	subset_size = file_size/fold_num
	accuracy = []

	for fold in xrange(fold_num):
		training_set = featuresets[:fold*subset_size] + featuresets[(fold+1)*subset_size:]
		testing_set = featuresets[fold*subset_size:][:subset_size]
		accuracy.append((nltk.classify.accuracy(classifier.train(training_set), testing_set))*100)
		# print fold+1, " Cross validation accuracy percent: ", (nltk.classify.accuracy(classifier.train(training_set), testing_set))*100
	return np.mean(accuracy)

def nb(featuresets):
	classifier = nltk.NaiveBayesClassifier
	accuracy = cv(featuresets, classifier)
	print "Naive Bayes Classifier accuracy percent: ", accuracy

	return accuracy

def lg(featuresets):
	'''Logistic Regression (aka logit, MaxEnt) classifier.'''
	classifier = SklearnClassifier(LogisticRegression())
	accuracy = cv(featuresets, classifier)
	print "Logistic Regression Classifier accuracy percent: ", accuracy

	return accuracy

def dt(featuresets):
	classifier = SklearnClassifier(DecisionTreeClassifier())
	accuracy = cv(featuresets, classifier)
	print "Decision Tree Classifier accuracy percent: ", accuracy

	return accuracy

def lsvm(featuresets):
	classifier = SklearnClassifier(LinearSVC())
	accuracy = cv(featuresets, classifier)
	print "Linear SVM classifier accuracy percent: ", accuracy

	return accuracy

# def rf(featuresets):
# 	classifier = SklearnClassifier(RandomForestClassifier())
# 	accuracy = cv(featuresets, classifier)
# 	print "Random Forest Classifier accuracy percent: ", accuracy

# 	return accuracy

# def svm(featuresets):
# 	classifier = SklearnClassifier(SVC())
# 	accuracy = cv(featuresets, classifier)
# 	print "SVM classifier accuracy percent: ", accuracy

# 	return accuracy





