Sentiment Classifier

LING 406 Intro to Computational Linguistics

University of Illinois at Urbana-Champaign

Cheng-Yang Peter Liu (cyliu4), 

May 8th, 2017


The code is run by Python 2.7 with numpy, scipy, sklearn, and NLTK libraries. 
You can change different experiment setting in the experiment.py file. 
This folder contain the following files:

1. experiment.py: main experiment file
	choose of setting:
		dataset: 0=movie reviews, 1=yelp reviews
		feature set: feat = [remove_stopwords, include_neg, keep_adjv, stemming]
2. read_data.py: imported by experiment.py, read movie reviews data, and Yelp’s restaurant data.
3. feature.py: imported by experiment.py, create Bag-of-Words feature, and add selected features.
4. classifier.py: imported by experiment.py, do four machine learning classification.
5. plot_feature_number.py: stand alone file, plot the accuracy along with different feature size.
6. README.txt
7. ling-406-term.pdf: project paper