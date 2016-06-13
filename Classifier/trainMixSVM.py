from sklearn import svm
import numpy as np
import random
from sklearn import preprocessing

def loadData(filename):
    f = open(filename)
    lines = f.readlines()
    f.close()
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
    return (data, label)

def getData(filename):
    data, label = loadData(filename)
    trainData, testData = data[:-1000], data[-1000:]
    trainLabel, testLabel = data[:-1000], data[-1000:]
    return trainData, testData, trainLabel, testLabel

def mixPredict(clfs, testData):
    results = []
    for clf in clfs:
        result = clf.decision_function(testData)
        results.append(result)
    clfCount = len(results)
    classResult = []
    sumResult = []
    for i in range(len(testData)):
        s = 0
        c = 0
        for r in range(len(results)):
            s += results[r][i]
            c += 1 if results[r][i] >= 0 else c -= 1
        sumResult.append(s)
        if c > 0:
            classResult.append(1)
        else if c < 0:
            classResult.append(0)
        else:
            # the number of positive and negative is the same
            # classify by the decision function
            if s >= 0:
                classResult.append(1)
            else:
                classResult.append(0)
    for s in sumResult:
        s = 1 if s >= 0 else s = 0
    return sumResult, classResult


if __name__ == '__main__':
    rgbTrainData, rgbTestData, trainLabel, testLabel = getData("color_clear_new_unify.txt")
    grayTrainData, grayTestData, _, _ = getData("gray_clear_new_unify.txt")
    hsvTrainData, hsvTestData, _, _ = getData("hsv_hist_new_unify.txt")
    labTrainData, labTestData, _, _ = getData("lab_hist_unify.txt")
    xyzTrainData, xyzTestData, _, _ = getData("xyz_hist_unify.txt")

    print 'begin train...'
    rgbClf = svm.SVC(kernel='rbf', gamma=g, C=c)
    rgbClf.fit(rgbTrainData, trainLabel)
    grayClf = svm.SVC(kernel='rbf', gamma=g, C=c)
    grayClf.fit(grayTrainData, trainLabel)
    hsvClf = svm.SVC(kernel='rbf', gamma=g, C=c)
    hsvClf.fit(hsvTrainData, trainLabel)
    labClf = svm.SVC(kernel='rbf', gamma=g, C=c)
    labClf.fit(labTrainData, trainLabel)
    xyzClf = svm.SVC(kernel='rbf', gamma=g, C=c)
    xyzClf.fit(xyzTrainData, trainLabel)
    print 'mix predict...'
    sumResult, classResult = mixPredict([rgbClf, hsvClf], testData)
    sumCount, classCount = 0.0, 0.0
    for i in range(len(testData)):
        if sumResult[i] == testLabel[i]:
            sumCount += 1
        if classResult[i] == testLabel[i]:
            classCount += 1
    print 'sum result : ' + str(sumCount / len(testData))
    print 'class result : ' + str(classCount / len(testData))