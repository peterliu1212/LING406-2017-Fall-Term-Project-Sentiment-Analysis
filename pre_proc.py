import nltk

def add_feature(documents, remove_stopwords, include_neg, keep_adjv, stemming):
	if remove_stopwords:
		i = 0
		for files, cats in documents:
			for sw in stopwords.words('english'):
				files = [w for w in files if w!=sw]
			documents[i] = tuple([files,cats])
			i += 1

	if include_neg:
		sentim_analyzer = SentimentAnalyzer()
		documents = [nltk.sentiment.util.mark_negation(doc) for doc in documents]

	if keep_adjv:
		i = 0
		for files, cats in documents:
			tagged = nltk.pos_tag(files)
			files = [w for w, tag, in zip(files,tagged) 
				 if re.search(r'^(JJ|RB)R?S?$',  tag[1])]
			documents[i] = tuple([files,cats])
			i += 1

	if stemming:
		ps = nltk.stem.PorterStemmer()
		i = 0
		for files, cats in documents:
			temp = []
			for w in files:
				if w == 'oed': # nltk bug: http://stackoverflow.com/questions/41517595/nltk-stemmer-string-index-out-of-range
					continue
				temp.append(ps.stem(w))
			documents[i] = tuple([temp, cats])
			i += 1
		# documents = [([ps.stem(w) for w in files], cats) for files, cats in documents ]

	return documents