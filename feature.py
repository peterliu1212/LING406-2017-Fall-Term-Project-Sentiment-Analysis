import nltk

def find_features(document, word_features):
	words = set(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features

def bow(documents, word_size):
	all_words = []
	for files in documents:
		for w in files[0]:
			all_words.append(w.lower())

	all_words = nltk.FreqDist(all_words)
	# print len(all_words.keys())
	word_features = list(all_words.keys())[:word_size]

	featuresets = [(find_features(rev, word_features), category) for (rev, category) in documents]

	return featuresets

def bowPOS(documents, word_size):
	all_words = []
	for files in documents:
		tagged = nltk.pos_tag(files[0])
		for w, tag in zip(files[0],tagged):
			all_words.append(w.lower() + '_' + tag[1])

	all_words = nltk.FreqDist(all_words)
	word_features = list(all_words.keys())[:word_size]

	featuresets = [(find_features(rev, word_features), category) for (rev, category) in documents]

	return featuresets



# def bigram(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
#     bigram_finder = BigramCollocationFinder.from_words(words)
#     bigrams = bigram_finder.nbest(score_fn, n)
#     return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
 
