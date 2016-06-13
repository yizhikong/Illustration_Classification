import os
import sys
import re

if __name__ == '__main__':
    filenames = os.listdir(os.getcwd())
    recordFiles = []
    for filename in filenames:
        if 'record' in filename and 'cnn' not in filename and '.txt' in filename:
            recordFiles.append(filename)
    for record in recordFiles:
        print record
        lines = open(record).readlines()
        gammaPat = re.compile('gamma=(-?\d\.\d+)')
        cPat = re.compile('C=(-?\d\.\d+)')
        msg = []
        index = 0
        while index < len(lines):
            item = {'gamma':gammaPat.search(lines[index]).group(1),
                    'C':cPat.search(lines[index]).group(1)}
            marks = map(float, lines[index+1].strip().split('\t'))
            #item['marks'] = marks
            item['aver'] = sum(marks) / len(marks)
            msg.append(item)
            index += 2
        print max(msg, key = lambda x : x['aver'])
