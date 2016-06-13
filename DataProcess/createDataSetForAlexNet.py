import cv2
import random
import os
import shutil

if __name__ == '__main__':
	top = open(r'D:\\code\\commicDownload\\top.txt').readlines()
	bottom = open(r'D:\\code\\commicDownload\\bottom.txt').readlines()
	random.shuffle(top)
	random.shuffle(bottom)
	trainData = top[:1000] + bottom[:1000]
	random.shuffle(trainData)
	valData = top[1000:1500] + bottom[1000:1500]
	random.shuffle(valData)
	os.mkdir('DataSet2')
	train = open(r'DataSet2\train.txt', 'w')
	for data in trainData:
		illust, mark, filename = data.strip().split('\t')
		newFileName = filename.split('\\')[-1]
		shutil.copy(filename, r'DataSet2\\' + newFileName)
		mark = '1' if int(mark) > 100 else '0'
		train.write(newFileName + '\t' + mark + '\n')
	train.close()
	val = open(r'DataSet2\val.txt', 'w')
	for data in valData:
		illust, mark, filename = data.strip().split('\t')
		newFileName = filename.split('\\')[-1]
		shutil.copy(filename, r'DataSet2\\' + newFileName)
		mark = '1' if int(mark) > 100 else '0'
		val.write(newFileName + '\t' + mark + '\n')
	val.close()

