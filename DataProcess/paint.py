import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	lines = open(r'D:\code\commicDownload\hsv_hist_new_unify_h.txt').readlines()
	for line in lines:
		if int(line.split(' ')[-1]) < 5000:
			continue
		data = map(lambda x : float(x), line.split('\t')[1].split(' ')[:180])
		plt.bar(np.arange(len(data)), data, alpha = 0.5)
		plt.show()