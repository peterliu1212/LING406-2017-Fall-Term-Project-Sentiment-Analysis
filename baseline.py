# Sentiment Classifier
# LING 406 Intro to Computational Linguistics
# University of Illinois at Urbana-Champaign
# Cheng-Yang Peter Liu (cyliu4), 
# May 8th, 2017
#
# This is the baseline from P. Bo, L. Lee, and S. Vaithyanathan
# "Thumbs up?: sentiment classification using machine learning techniques."

import nltk
import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories()
			 for fileid in movie_reviews.fileids(category)]

pos_words = ["love", "wonderful", "best", "great", "superb", "still", "beautiful"]
neg_words = ["bad", "worst", "stupid", "waste", "boring", "?", "!"]

num_file = len(documents)
frac = 0
frac_pos = 0
frac_neg = 0
for file in range(num_file):
	current_doc = documents[file]
	current_label = current_doc[1]
	current_words = []
	pos_count = 0
	neg_count = 0
	for w in current_doc[0]:
		if w in pos_words:
			pos_count += 1
		if w in neg_words:
			neg_count += 1
	if pos_count >= neg_count:
		current_result = 'pos'
	else:
		current_result = 'neg'

	# if current_result == current_label:
	# 	# print "correct!"
	# 	frac += 1
	if current_result == 'pos' and current_label == 'pos':
		frac_pos += 1
	if current_result == 'neg' and current_label == 'neg':
		frac_neg += 1

accuracy = (float(frac_neg+frac_pos)/float(num_file)) * 100
accuracy_pos = (float(frac_pos)/float(num_file/2)) * 100
accuracy_neg = (float(frac_neg)/float(num_file/2)) * 100

print "accuracy: ", accuracy
print "accuracy_pos: ", accuracy_pos
print "accuracy_neg: ", accuracy_neg
