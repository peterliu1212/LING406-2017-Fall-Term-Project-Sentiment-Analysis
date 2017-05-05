import re
import matplotlib.pyplot as plt

acy_nb = []
acy_lg = []
acy_dt = []
acy_svm = []
parameter = []

dataset = 0 # movie review
# dataset = 1 # yelp review

with open('movie_bow_100_30000.txt') as f:
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

plt.plot(parameter, acy_nb, 'bo')
p1, = plt.plot(parameter, acy_nb, 'b')
plt.plot(parameter, acy_lg, 'ko')
p2, = plt.plot(parameter, acy_lg, 'k')
plt.plot(parameter, acy_dt, 'ro')
p3, = plt.plot(parameter, acy_dt, 'r')
plt.plot(parameter, acy_svm, 'go')
p4, = plt.plot(parameter, acy_svm, 'g')
plt.legend([p1, p2, p3, p4], ['Naive Bayes', 'Logistic Regression', 'Decision Tree', 'Support Vector Machine'])
plt.axis([0, 30000, 0, 100])
plt.ylabel('Accuracy')
plt.xlabel('Feature Size')
plt.title('Baseline: Bag of Word feature')
plt.show()