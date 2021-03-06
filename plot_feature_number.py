# Sentiment Classifier
# LING 406 Intro to Computational Linguistics
# University of Illinois at Urbana-Champaign
# Cheng-Yang Peter Liu (cyliu4), 
# May 8th, 2017
#
# This file plot the accuracy with different feature size

import re
import matplotlib.pyplot as plt

def read_acy(file_name):
	# read the saved accuracy files
	acy_nb = []
	acy_lg = []
	acy_dt = []
	acy_svm = []
	parameter = []

	with open(file_name) as f: # open the files and read
		for line in f:
			data = line.split()
			para = re.findall(r'^[0-9][0-9][0-9][0-9]?[0-9]?', data[0])
			nb = re.findall(r'^[0-9][0-9][.][0-9][0-9]?', data[1])
			lg = re.findall(r'^[0-9][0-9][.][0-9][0-9]?', data[2])
			dt = re.findall(r'^[0-9][0-9][.][0-9][0-9]?', data[3])
			svm = re.findall(r'^[0-9][0-9][.][0-9][0-9]?', data[4])

			acy_nb.append(float(''.join(nb)))
			acy_lg.append(float(''.join(lg)))
			acy_dt.append(float(''.join(dt)))
			acy_svm.append(float(''.join(svm)))
			parameter.append(int(''.join(para)))


	return [parameter, acy_nb, acy_lg, acy_dt, acy_svm]

def plot_acy(acy, title):
	# plot the accuracy
	plt.plot(acy[0], acy[1], 'bo')
	p1, = plt.plot(acy[0], acy[1], 'b')
	plt.plot(acy[0], acy[2], 'ko')
	p2, = plt.plot(acy[0], acy[2], 'k')
	plt.plot(acy[0], acy[3], 'ro')
	p3, = plt.plot(acy[0], acy[3], 'r')
	plt.plot(acy[0], acy[4], 'go')
	p4, = plt.plot(acy[0], acy[4], 'g')
	plt.legend([p1, p2, p3, p4], ['Naive Bayes', 'Logistic Regression', 'Decision Tree', 'Support Vector Machine'])
	plt.axis([0, 10000, 0, 100])
	plt.ylabel('Accuracy')
	plt.xlabel('Feature Size')
	plt.title(title)
	plt.show()

# Read the data, and plot the data
file_movie_bow = 'movie_bow_100_30000.txt'
acy_movie_bow = read_acy(file_movie_bow)
title_movie_bow = 'Movie Baseline: Bag of Word feature'
plot_acy(acy_movie_bow, title_movie_bow)

file_movie_bow = 'movie_bowRS_100_30000.txt'
acy_movie_bow = read_acy(file_movie_bow)
title_movie_bow = 'Movie: Bag of Word feature + Remove Stopwords'
plot_acy(acy_movie_bow, title_movie_bow)

file_movie_bow = 'movie_bowNeg_100_30000.txt'
acy_movie_bow = read_acy(file_movie_bow)
title_movie_bow = 'Movie: Bag of Word feature + Include Negation'
plot_acy(acy_movie_bow, title_movie_bow)

file_movie_bow = 'movie_bowADJV_100_30000.txt'
acy_movie_bow = read_acy(file_movie_bow)
title_movie_bow = 'Movie: Bag of Word feature + Adjectives and Adverbs'
plot_acy(acy_movie_bow, title_movie_bow)

file_movie_bow = 'movie_bowStem_100_30000.txt'
acy_movie_bow = read_acy(file_movie_bow)
title_movie_bow = 'Movie: Bag of Word feature + Remove Stopwords'
plot_acy(acy_movie_bow, title_movie_bow)

file_movie_bow = 'yelp_bow_100_30000.txt'
acy_movie_bow = read_acy(file_movie_bow)
title_movie_bow = 'Yelp Baseline: Bag of Word feature'
plot_acy(acy_movie_bow, title_movie_bow)

file_movie_bow = 'yelp_bowRS_100_30000.txt'
acy_movie_bow = read_acy(file_movie_bow)
title_movie_bow = 'Yelp: Bag of Word feature + Remove Stopwords'
plot_acy(acy_movie_bow, title_movie_bow)

file_movie_bow = 'yelp_bowNeg_100_30000.txt'
acy_movie_bow = read_acy(file_movie_bow)
title_movie_bow = 'Yelp: Bag of Word feature + Include Negation'
plot_acy(acy_movie_bow, title_movie_bow)

file_movie_bow = 'yelp_bowADJV_100_30000.txt'
acy_movie_bow = read_acy(file_movie_bow)
title_movie_bow = 'Yelp: Bag of Word feature + Adjectives and Adverbs'
plot_acy(acy_movie_bow, title_movie_bow)

file_movie_bow = 'yelp_bowStem_100_30000.txt'
acy_movie_bow = read_acy(file_movie_bow)
title_movie_bow = 'Yelp: Bag of Word feature + Remove Stopwords'
plot_acy(acy_movie_bow, title_movie_bow)



