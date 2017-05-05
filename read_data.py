from nltk.corpus import movie_reviews
from nltk.tokenize import TweetTokenizer
import re
import os.path
import pickle


def movie():
	if os.path.exists('movie_doc.pickle'):
		save_doc = open('movie_doc.pickle', 'rb')
		documents = pickle.load(save_doc)
		save_doc.close()
	else:
		documents = [(list(movie_reviews.words(fileid)), category) 
				 for category in movie_reviews.categories()
				 for fileid in movie_reviews.fileids(category)]
		save_doc = open('movie_doc.pickle', 'wb')
		pickle.dump(documents, save_doc)
		save_doc.close()
	return documents

def yelp():
	print 'Start input Yelp review data.'
	if os.path.exists('yelp_doc.pickle'):
		save_doc = open('yelp_doc.pickle', 'rb')
		documents = pickle.load(save_doc)
		save_doc.close()
	else:
		fname = 'yelp_reviews/all_reviews.txt'

		with open(fname) as f:
			files = f.readlines()


		# you may also want to remove whitespace characters like `\n` at the end of each line
		files = [x.strip() for x in files] 

		content = []
		star = []
		star_flag = False
		temp = []
		for text in files:
			if text == '{{{':
				star_flag = True
				continue
			if text == '}}}':
				continue
			if text == '[[[':
				continue
			if star_flag == True:
				star.append(re.findall(r'^[1-5]',text)[0])
				star_flag = False
				temp = []
				continue
			else:
				if text == ']]]':
					content.append(temp)
					continue
				else:
					temp.append(text)

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

		documents = [(words, category) for words, category in zip(data,star)]
		save_doc = open('yelp_doc.pickle', 'wb')
		pickle.dump(documents, save_doc)
		save_doc.close()
	return documents
