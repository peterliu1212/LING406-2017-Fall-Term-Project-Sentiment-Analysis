# Sentiment Classifier
# LING 406 Intro to Computational Linguistics
# University of Illinois at Urbana-Champaign
# Cheng-Yang Peter Liu (cyliu4), 
# May 8th, 2017
#
# This file includes two functions to read two different dataset.
# 1. Movie reviews, http://www.cs.cornell.edu/people/pabo/movie-review-data/, 
#    Pang et al (2002).
# 2. Yelp's Champaign-Urbana area restaurant reviews, originally created by 
#    John Hall, in LING 406 Spring 2016. 

from nltk.corpus import movie_reviews
from nltk.tokenize import TweetTokenizer
import re
import os.path
import pickle


def movie():
	print 'Start input movie reviews data.'
	# check if we have saved dataset
	if os.path.exists('movie_doc.pickle'): 
		save_doc = open('movie_doc.pickle', 'rb')
		documents = pickle.load(save_doc) # load the dataset
		save_doc.close()
	else:
		# If we don't have saved dataset, create one
		# The movie reviews dataset is generate from the nltk library
		documents = [(list(movie_reviews.words(fileid)), category) 
				 for category in movie_reviews.categories()
				 for fileid in movie_reviews.fileids(category)]
		save_doc = open('movie_doc.pickle', 'wb')
		pickle.dump(documents, save_doc) # save the dataset for the future use
		save_doc.close()
	return documents

def yelp():
	print 'Start input Yelp reviews data.'
	# check if we have saved dataset
	if os.path.exists('yelp_doc.pickle'):
		save_doc = open('yelp_doc.pickle', 'rb')
		documents = pickle.load(save_doc)
		save_doc.close()
	else:
		# If we don't have saved dataset, create one
		fname = 'yelp_reviews/all_reviews.txt' # read the data from the file
		with open(fname) as f:
			files = f.readlines() # read the data line by line
		files = [x.strip() for x in files] # seperate each line

		content = []
		star = []
		star_flag = False
		temp = []
		# Process the data into desire format
		for text in files:
			if text == '{{{':
				# find the start of star label
				star_flag = True
				continue
			if text == '}}}':
				# find the end of star label
				continue
			if text == '[[[':
				# find the start of text
				continue
			if star_flag == True:
				# read the stars
				star.append(re.findall(r'^[1-5]',text)[0])
				star_flag = False
				temp = []
				continue
			else:
				if text == ']]]':
					# find the end of text, and read the text
					content.append(temp)
					continue
				else:
					temp.append(text)

		# tokenize the words
		tknzr = TweetTokenizer()
		data = []
		for files in content:
			temp = []
			i = 0
			num = len(files)
			for x in files:
				if 3 < i < num-3:
					temp.append(tknzr.tokenize(x))
				i += 1
			data.append([(words) for lines in temp for words in lines])
		# Generate the data format
		documents = [(words, category) for words, category in zip(data,star)]
		save_doc = open('yelp_doc.pickle', 'wb')
		pickle.dump(documents, save_doc) # save the dataset for the future use
		save_doc.close()
	return documents
