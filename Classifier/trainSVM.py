from sklearn import svm
import numpy as np
import random
import sys
from sklearn import preprocessing

def getWeight(label):
    weight = {}
    for l in label:
        if l not in weight:
            weight[l] = 1
        else:
            weight[l] += 1
    print weight
    for l in weight:
        weight[l] /= float(len(label))
    print weight
    return weight

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
    weight = getWeight(label)
    return (data, label, weight)

def saveClassifier(gamma, c):
    data, label, weight = loadData()
    #data = preprocessing.scale(data)
    print data.shape
    trainData, testData = data[:-20000], data[-20000:]
    trainLabel, testLabel = label[:-20000], label[-20000:]
    print trainData.shape
    print trainLabel.shape
    print 'begin train...'
    clf = svm.SVC(kernel='rbf', gamma=gamma, C=c)
    clf.fit(trainData, trainLabel)
    print 'predict...'
    res = clf.predict(testData)
    print 'show probability'
    print clf.decision_function(testData[:20])
    count = 0.0
    print res[:20]
    print sum(res)
    print testLabel[:20]
    for i in range(len(testData)):
        if res[i] == testLabel[i]:
            count += 1
    print count / len(testData)
    result = 'gamma=%f, C=%f, accuracy=%f' % (gamma, c, count / len(testData))
    print result
    cPickle.dump(clf, open("classifier.pkl", "w"))

if __name__ == '__main__':
    dataFile = r'D:\code\commicDownload\hsv_hist_new_unify_merge4_sort_smooth3.txt'
    recordFile = 'record_hsv_hist_new_unify_merge4_sort_smooth3_full.txt'
    print sys.argv
    if len(sys.argv) > 2:
        dataFile = sys.argv[1]
        recordFile = sys.argv[2]
    parameter_gamma = [0.001, 0.01, 0.1, 1, 10]
    parameter_c = [1, 10]
    print parameter_gamma
    print parameter_c
    f = open(recordFile, 'a')
    for g in parameter_gamma:
        for c in parameter_c:
            f.write('gamma=%f, C=%f:\n' % (g, c))
            for i in range(3):
                data, label, weight = loadData()
                #data = preprocessing.scale(data)
                print data.shape
                trainData, testData = data[:-1000], data[-1000:]
                trainLabel, testLabel = label[:-1000], label[-1000:]
                print trainData.shape
                print trainLabel.shape
                print 'begin train...'
                clf = svm.SVC(kernel='rbf', gamma=g, C=c)
                clf.fit(trainData, trainLabel)
                print 'predict...'
                res = clf.predict(testData)
                print 'show probability'
                print clf.decision_function(testData[:20])
                count = 0.0
                print res[:20]
                print sum(res)
                print testLabel[:20]
                for i in range(len(testData)):
                    if res[i] == testLabel[i]:
                        count += 1
                print count / len(testData)
                result = 'gamma=%f, C=%f, accuracy=%f' % (g, c, count / len(testData))
                print result
                f.write(str(count / len(testData)) + '\t')
            f.write('\n')
            f.flush()
    f.close()
