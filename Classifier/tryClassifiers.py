from sklearn import svm
import numpy as np
import random
import sys
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors


def loadData():
    #filenames = os.listdir(os.getcwd())
    filenames = [dataFile]
    for filename in filenames:
        if 'txt' in filename and 'sum' not in filename:
            f = open(filename)
    lines = f.readlines()
    f.close()
    random.shuffle(lines)
    data = []
    label = []
    for i in range(len(lines)):
        line = lines[i][:]
        lines[i] = ''
        pos = line.find(' ')
        if pos < 0:
            continue
        line = line[pos+1 :].strip()
        spLine = line.split(' ')
        if int(spLine[-1]) < 100:
            spLine[-1] = 0
        elif int(spLine[-1]) > 100:
            spLine[-1] = 1
        else:
            continue
        data.append(spLine[:-1])
        label.append(spLine[-1])
    print 'array...'
    data = np.array(data, dtype = float)
    label = np.array(label, dtype = int)
    print 'score...'
    return (data, label)

if __name__ == '__main__':
    dataFile = r'hsl_hist_unify.txt'
    recordFile = 'record_hsl_hist_unify.txt'
    data, label = loadData()
    data = preprocessing.scale(data)
    print data.shape
    trainData, testData = data[:-10000], data[-10000:]
    trainLabel, testLabel = label[:-10000], label[-10000:]
    print trainData.shape
    print trainLabel.shape
    print 'begin train...'
    #clf = svm.LinearSVC()
    #clf = RandomForestClassifier(n_estimators=64)
    #clf = DecisionTreeClassifier(max_depth=None)
    for i in range(3, 10):
        clf = neighbors.KNeighborsClassifier(i, weights='distance')
        clf.fit(trainData, trainLabel)
        print 'predict...' + str(i) + 'nn...'
        res = clf.predict(testData)
        count = 0.0
        print res[:20]
        res = map(lambda x : 1 if x > 0.5 else 0, res)
        print res[:20]
        print sum(res)
        print testLabel[:20]
        for i in range(len(testData)):
            if res[i] == testLabel[i]:
                count += 1
        print count / len(testData)