import os
import cv2

if __name__ == '__main__':
    fileList = os.listdir(os.getcwd())
    msgTps = []
    for f in fileList:
        if 'jpg' not in f:
            continue
        if 'threshold' in f:
            os.remove(f)
        msg = f.split('.')[0].split('_')
        print msg
        if len(msg) != 4:
            continue
        iid, view, count, score = msg
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        ret, img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
        #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
        saveName = f.split('.')[0] + '_binary.' + f.split('.')[1]
        cv2.imwrite(saveName, img)
        # print msg
        per1 = float(score) / (int(count) + 0.0001)
        per2 = float(score) / int(view)
        msgTps.append((per1, per2, msg))
    for tp in msgTps:
        print tp
    
