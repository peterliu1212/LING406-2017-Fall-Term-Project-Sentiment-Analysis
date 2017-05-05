import re
import matplotlib.pyplot as plt

accuracy = []
parameter = []

with open('para.txt') as f:
	for line in f:
		data = line.split()
		num = re.findall(r'^[0-9][0-9][.][0-9][0-9]?', data[0])
		para = re.findall(r'^[0-9][0-9]?[0-9]?[0-9]?[0-9]?', data[1])
		
		accuracy.append(float(''.join(num)))
		parameter.append(int(''.join(para)))

plt.plot(parameter, accuracy, 'bo')
plt.plot(parameter, accuracy, 'k')
plt.axis([0, 40000, 0, 100])
plt.ylabel('Accuracy')
plt.xlabel('Feature Size')
plt.show()