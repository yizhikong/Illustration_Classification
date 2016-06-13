import cv2
import random
import os
import shutil
import cPickle
from sklearn import svm
import numpy as np
from imgDict import adjustSize, getImageHist
import quantize

def generateHSVFeature(imgName):
	img = cv2.imread(imgName)
	if img is None:
		return None
	img = adjustSize(img)
	if img is None:
		return None
	b, g, r = cv2.split(img)
	if b.tolist() == g.tolist() and g.tolist() == r.tolist():
		print 'gray image'
		return None
	img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	height = img.shape[0]
	width = img.shape[1]
	hist = getImageHist(img, [180, 256, 256], [180.0, 256.0, 256.0])
	hist = map(lambda x : round(float(x)/(height*width), 3), hist)
	return hist

def hsvQuantize(nums, hstep, svstep):
    nums = map(lambda x : float(x), nums)
    i = 0
    color = 180 / hstep
    changeCount = 0
    result = []
    step = hstep
    while i < len(nums):
        result.append(sum(nums[i:i+step]))
        i += step
        changeCount += 1
        if changeCount == color:
            step = svstep
    return result

def quantize(level, nums):
    nums = map(lambda x : float(x), nums)
    step = 256 / level
    i = 0
    result = []
    while i < len(nums):
        result.append(sum(nums[i:i+step]))
        i += step
    return result

def getInputData(imgName):
	hist = generateHSVFeature(imgName)
	if hist == None:
		return None
	hist = quantize(128, hist[1:])[1:]
	na = np.array(hist).reshape(1, len(hist))
	return na

if __name__ == '__main__':
	top = open(r'D:\\code\\commicDownload\\top.txt').readlines()
	bottom = open(r'D:\\code\\commicDownload\\bottom.txt').readlines()
	random.shuffle(top)
	random.shuffle(bottom)
	analyzeData = top[:3000] + bottom[:3000]
	random.shuffle(analyzeData)
	clf = cPickle.load(open('hsv_hist_90C_classifier.pkl'))
	os.mkdir('CorrectPositive')
	os.mkdir('CorrectNegative')
	os.mkdir('Positive2Negative')
	os.mkdir('Negative2Positive')
	correct = 0
	for data in analyzeData:
		illust, mark, filename = data.strip().split('\t')
		newFileName = filename.split('\\')[-1]
		test = getInputData(filename)
		if test == None:
			continue
		res = clf.predict(test)
		mark = int(mark)
		if mark > 100:
			if res[0] == 1:
				shutil.copy(filename, r'CorrectPositive\\' + newFileName)
				correct += 1
			else:
				shutil.copy(filename, r'Positive2Negative\\' + newFileName)
		else:
			if res[0] == 0:
				shutil.copy(filename, r'CorrectNegative\\' + newFileName)
				correct += 1
			else:
				shutil.copy(filename, r'Negative2Positive\\' + newFileName)
	print float(correct) / len(analyzeData)