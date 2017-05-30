import numpy as np
import matplotlib.pyplot as plt

def processMethodResult(path):
	print "Analysing Method %s" % path

	avg = {'p':[], 'r':[]}
	with open(path) as fp:
		for line in fp:
			if line.startswith("#"):
				continue            
            
			aLine = line.split(" ")
			values = aLine[1:-1]
			pr = ([], [])
			c = 0
			for i in values:
				if c % 2 == 0:
					pr[0].append(i)
				else:
					pr[1].append(i)

				c = c + 1

			avg['p'].append(np.average(np.array(pr[0]).astype(float)))
			avg['r'].append(np.average(np.array(pr[1]).astype(float)))
	return avg

def plot(x1, x2):
	plt.clf()
	plt.plot(x1["r"], x1["p"], color='gold', lw=2, label='method 1')
	plt.plot(x2["r"], x2["p"], color='blue', lw=2, label='method 2')

	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall curves')
	plt.show()

if __name__ == "__main__":
	print "Test"
	
	m1_avgs = processMethodResult('method1/result/result_all.txt')
	m2_avgs = processMethodResult('method2/result/result_all.txt')

	plot(m1_avgs, m2_avgs)
