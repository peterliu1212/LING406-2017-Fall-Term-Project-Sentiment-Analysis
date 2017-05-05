import nltk
import random

import pre_proc
import classifier
import feature
import read_data

def tune(documents, feat, save_file, tune_para):
	file = open(save_file, 'w')
	acy = []
	for parameter in tune_para:
		print 'parameter: ', parameter
		if feat == 0:
			featuresets = feature.bow(documents, parameter)
		elif feat == 1:
			featuresets = feature.bowPOS(documents, parameter)

		# Classification
		print 'Start Naive Bayes Classification'
		nb_result = classifier.nb(featuresets)
		print 'Start Logistic Regression Classification'
		lg_result = classifier.lg(featuresets)
		print 'Start Decision Tree Classification'
		dt_result = classifier.dt(featuresets)
		print 'Start SVM Classification'
		lsvm_result = classifier.lsvm(featuresets)

		acy.append([nb_result, lg_result, dt_result, lsvm_result])
		file.write("%s, %s, %s, %s, %s\n" % (parameter, nb_result, lg_result, dt_result, lsvm_result))

	file.close()

# Choose dataset
# dataset = 0 # movie review
dataset = 1 # yelp review
# Read data and preprocessing
pre = [False, False, False, False] # pre = [remove_stopwords, include_neg, keep_adjv, stemming]
if dataset == 0:
	documents = read_data.movie() # input data
	save_dataset = 'movie'
else:
	documents = read_data.yelp()
	save_dataset = 'yelp'
documents = pre_proc.add_feature(documents, pre[0], pre[1], pre[2], pre[3]) # preprocessing
random.shuffle(documents)

# Run Experiment 1
print 'Experiment 1: Baseline, Bag of Word'
# Generate feature sets
para = range(100,1000,200) + range(1000,10000,2000) + range(10000, 40000, 10000)# parameter size
# Feature selection
feat = 0 # 0:BoW, 1:BoW + POS, 2: 
# save result file name
save_feature = 'bow'
save_para = '100_30000'
save_file = save_dataset + '_' + save_feature + '_' + save_para + '.txt'
# run experiment
para_bow = tune(documents, feat, save_file, para)

# # Run Experiment 2
# print 'Experiment 2: Bag of Word + POS'
# # Generate feature sets
# para = range(100,1000,200) + range(1000,10000,2000) + range(10000, 40000, 10000)# parameter size
# # Feature selection
# feat = 1 # 0:BoW, 1:BoW + POS, 2: 
# # save result file name
# save_feature = 'bowPOS'
# save_dataset = 'movie'
# save_para = '100_30000'
# save_file = save_dataset + '_' + save_feature + '_' + save_para + '.txt'
# # run experiment
# para_bow = tune(documents, feat, save_file, para)