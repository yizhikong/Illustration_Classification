import os
import numpy as np
import cv2
import random
from numba import jit

imgs = {}
rootPath = 'D:\code\commicDownload'

def loadImgs(rootpath):
    """load images using dfs."""
    global imgs
    filenames = os.listdir(rootpath)
    nextpaths = []
    for filename in filenames:
        if '.' not in filename:
            nextpaths.append(rootpath + '\\' + filename)
        else:
            if 'txt' in filename:
                continue
            msg = filename.split('.')[0].split('_')
            if len(msg) == 4:
                iid, view, count, score = msg
            elif len(msg) == 5:
                rank, view, count, score, iid = msg
            else:
                continue
            if iid in imgs and imgs[iid]['score'] > (int)(score):
                continue
            else:
                imgs[iid] = {'score' : (int)(score),
                             'filename' : rootpath + '\\' + filename}
    for path in nextpaths:
        loadImgs(path)

def slip(imgs):
    """slip the images to get two rank lists(bottom, top)"""
    sortedImgs = sorted(imgs.iteritems(), key = lambda x : x[1]['score'])
    count = 0
    top = []
    for img in sortedImgs:
        if 'rankDown' in img[1]['filename']:
            top.append(img)
    count = len(top)
    print 'size' + repr(count)
    bottom = sortedImgs[:count]
    print top[:10]
    f = open(rootPath + '\\' + 'bottom.txt', 'w')
    for b in bottom:
        f.write(b[0] + '\t' + repr(b[1]['score']) + '\t' + b[1]['filename'] + '\n')
    f.close()
    f = open(rootPath + '\\' + 'top.txt', 'w')
    for t in top:
        f.write(t[0] + '\t' + repr(t[1]['score']) + '\t' + t[1]['filename'] + '\n')
    f.close()
    return bottom, top

def adjustSize(img):
    """adjust the size of image"""
    height = img.shape[0]
    width = img.shape[1]
    if height / width > 2 or width / height > 2:
        return None
    nw, nh, scale = 0, 0, 1
    if height >= width:
        nh, nw = 708, 500
    else:
        nw, nh = 500, 708
    '''
    if height > width and width > 500:
        scale = 500.0 / width
    if height < width and height > 500:
        scale = 500.0 / height
    nw = (int)(width * scale)
    nh = (int)(height * scale)
    '''
    nw, nh = 560, 560
    img = cv2.resize(img, (nw, nh))
    return img

def getHistogramFromLayer(layer, level=256, limit=256.0):
    # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]])
    hist = cv2.calcHist([layer], [0], None, [level], [0.0, limit])
    return flattern(hist)
    #b, g, r = cv2.split(img)

def getImageHist(img, level=[256,256,256], limit=[256.0,256.0,256.0]):
    b, g, r = cv2.split(img)
    hist = getHistogramFromLayer(b, level[0], limit[0]) + getHistogramFromLayer(g, level[1], limit[1]) + getHistogramFromLayer(r,level[2], limit[2])
    return hist

def flattern(listOfList):
    l = []
    for subList in listOfList:
        l.append((int)(subList[0]))
    return l

def countSum(hist):
    s = 0
    for i in range(256):
        s += i * hist[i]
    return s

def write(index, fop, hist, point):
    fop.write('index:' + str(index) + '\t')
    for h in hist:
        fop.write(str(h) + ' ')
    fop.write(point + '\n')

def getPoint(imgName):
    """Get the point of the image from image name"""
    name = imgName.split('\\')[-1]
    msg = name.split('.')[0].split('_')
    if len(msg) == 4:
        return msg[-1]
    elif len(msg) == 5:
        return msg[-2]

def generateFromRGB(imgNames):
    colorF = open(rootPath + '\\' + "color_clear_new_unify.txt", "a")
    grayF = open(rootPath + '\\' + "gray_clear_new_unify.txt", "a")
    count = 0
    process = 0
    for imgName in imgNames:
        print str(process) + '/' + str(len(imgNames))
        process += 1
        point = getPoint(imgName)
        img = cv2.imread(imgName)
        if img is None:
            continue
        img = adjustSize(img)
        if img is None:
            continue
        b, g, r = cv2.split(img)
        if b.tolist() == g.tolist() and g.tolist() == r.tolist():
            print 'gray image'
            continue
        height = img.shape[0]
        width = img.shape[1]
        colorHist = getImageHist(img)
        colorHist = map(lambda x : round(float(x)/(height*width), 3), colorHist)
        if len(colorHist) == 0:
            print 'unexpected error!'
            continue
        write(process, colorF, colorHist, point)
        grayHist = getHistogramFromLayer(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        grayHist = map(lambda x : round(float(x)/(height*width), 3), grayHist)
        write(process, grayF, grayHist, point)
        count += 1
        if count > 200:
            colorF.flush()
            grayF.flush()
            count = 0
    colorF.close()
    grayF.close()

def generateDifHistFromHSV(imgNames):
    hsvdifF = open(rootPath + '\\' + "hsvdif_hist_new_unify.txt", "a")
    difF = open(rootPath + '\\' + "dif_hist_new_unify.txt", "a")
    count = 0
    process = 0
    for imgName in imgNames:
        print str(process) + '/' + str(len(imgNames))
        process += 1
        point = getPoint(imgName)
        img = cv2.imread(imgName)
        if img is None:
            continue
        img = adjustSize(img)
        if img is None:
            continue
        b, g, r = cv2.split(img)
        if b.tolist() == g.tolist() and g.tolist() == r.tolist():
            print 'gray image'
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height = img.shape[0]
        width = img.shape[1]
        hist = getImageHist(img, [180, 256, 256], [180.0, 256.0, 256.0])
        hist = map(lambda x : round(float(x)/(height*width), 3), hist)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dif = gray % 32
        difHist = getHistogramFromLayer(dif, level=32, limit=32.0)
        difHist = map(lambda x : round(float(x)/(height*width), 3), difHist)
        hist += difHist
        if len(hist) == 0:
            print 'unexpected error!'
            continue
        write(process, hsvdifF, hist, point)
        write(process, difF, difHist, point)
        count += 1
        if count > 200:
            hsvdifF.flush()
            difF.flush()
            count = 0
    hsvdifF.close()
    difF.close()

def generateFromDifImg(imgNames):
    difF = open(rootPath + '\\' + "dif_img.txt", "a")
    count = 0
    process = 0
    for imgName in imgNames:
        print str(process) + '/' + str(len(imgNames))
        process += 1
        point = getPoint(imgName)
        img = cv2.imread(imgName)
        if img is None:
            continue
        img = cv2.resize(img, (50, 50))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img % 32) / 32.0
        hist = img[:,:].flatten()
        height = img.shape[0]
        width = img.shape[1]
        write(process, difF, hist, point)
        count += 1
        if count > 200:
            difF.flush()
            count = 0
    difF.close()

def generateSumImg(imgNames):
    difF = open(rootPath + '\\' + "col_row_sum.txt", "a")
    count = 0
    process = 0
    for imgName in imgNames:
        print str(process) + '/' + str(len(imgNames))
        process += 1
        point = getPoint(imgName)
        img = cv2.imread(imgName)
        if img is None:
            continue
        img = cv2.resize(img, (200, 200))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        vimg = img[1:]
        vimg = np.vstack((vimg, np.array([0] * vimg.shape[1])))
        vabs = abs((img - vimg)[:-1])
        vabs = np.rollaxis(vabs, 1)
        vsum = map(sum, vabs)

        himg = img[:,1:]
        img = img[:, :-1]
        habs = abs(img - himg)
        hsum = map(sum, habs)

        hist = vsum + hsum
        write(process, difF, hist, point)
        count += 1
        if count > 200:
            difF.flush()
            count = 0
    difF.close()

def filtPixiv(img):
    newImg = []
    height, width, _ = img.shape
    for h in range(height):
        for w in range(width):
            luma = sum(img[h][w]) / 3.0
            if luma > 245 or luma < 10:
                continue
            else:
                newImg.append(img[h][w])
    for i in range(2, 100):
        if len(newImg) % i == 0:
            newImg = np.array(newImg).reshape(len(newImg) / i, i, 3)
            return newImg
    return img


def generateFiltHistFromHSV(imgNames):
    hsvF = open(rootPath + '\\' + "hsv_filt_hist_new_unify.txt", "a")
    count = 0
    process = 0
    for imgName in imgNames:
        print str(process) + '/' + str(len(imgNames))
        process += 1
        point = getPoint(imgName)
        img = cv2.imread(imgName)
        if img is None:
            continue
        img = adjustSize(img)
        if img is None:
            continue
        b, g, r = cv2.split(img)
        if b.tolist() == g.tolist() and g.tolist() == r.tolist():
            print 'gray image'
            continue
        img = filtPixiv(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height = img.shape[0]
        width = img.shape[1]
        hist = getImageHist(img, [180, 256, 256], [180.0, 256.0, 256.0])
        hist = map(lambda x : round(float(x)/(height*width), 3), hist)
        if len(hist) == 0:
            print 'unexpected error!'
            continue
        write(process, hsvF, hist, point)
        count += 1
        if count > 200:
            hsvF.flush()
            count = 0
    hsvF.close()

def generateHistFromHSV(imgNames):
    hsvF = open(rootPath + '\\' + "hsv_hist_new_unify.txt", "a")
    count = 0
    process = 0
    for imgName in imgNames:
        print str(process) + '/' + str(len(imgNames))
        process += 1
        point = getPoint(imgName)
        img = cv2.imread(imgName)
        if img is None:
            continue
        img = adjustSize(img)
        if img is None:
            continue
        b, g, r = cv2.split(img)
        if b.tolist() == g.tolist() and g.tolist() == r.tolist():
            print 'gray image'
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height = img.shape[0]
        width = img.shape[1]
        hist = getImageHist(img, [180, 256, 256], [180.0, 256.0, 256.0])
        hist = map(lambda x : round(float(x)/(height*width), 3), hist)
        if len(hist) == 0:
            print 'unexpected error!'
            continue
        write(process, hsvF, hist, point)
        count += 1
        if count > 200:
            hsvF.flush()
            count = 0
    hsvF.close()

def generateHistFromHLS(imgNames):
    hlsF = open(rootPath + '\\' + "hls_hist_unify.txt", "a")
    count = 0
    process = 0
    for imgName in imgNames:
        print str(process) + '/' + str(len(imgNames))
        process += 1
        point = getPoint(imgName)
        img = cv2.imread(imgName)
        if img is None:
            continue
        img = adjustSize(img)
        if img is None:
            continue
        b, g, r = cv2.split(img)
        if b.tolist() == g.tolist() and g.tolist() == r.tolist():
            print 'gray image'
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        height = img.shape[0]
        width = img.shape[1]
        hist = getImageHist(img, [180, 256, 256], [180.0, 256.0, 256.0])
        hist = map(lambda x : round(float(x)/(height*width), 3), hist)
        if len(hist) == 0:
            print 'unexpected error!'
            continue
        write(process, hlsF, hist, point)
        count += 1
        if count > 200:
            hlsF.flush()
            count = 0
    hlsF.close()

def generateHistFromHLL(imgNames):
    hlLF = open(rootPath + '\\' + "hll_hist_unify.txt", "a")
    count = 0
    process = 0
    for imgName in imgNames:
        print str(process) + '/' + str(len(imgNames))
        process += 1
        point = getPoint(imgName)
        img = cv2.imread(imgName)
        if img is None:
            continue
        img = adjustSize(img)
        if img is None:
            continue
        b, g, r = cv2.split(img)
        if b.tolist() == g.tolist() and g.tolist() == r.tolist():
            print 'gray image'
            continue
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        height = img.shape[0]
        width = img.shape[1]
        hist = getHistogramFromLayer(h, 180, 180.0) + getHistogramFromLayer(l) + getHistogramFromLayer(L)
        hist = map(lambda x : round(float(x)/(height*width), 3), hist)
        if len(hist) == 0:
            print 'unexpected error!'
            continue
        write(process, hlLF, hist, point)
        count += 1
        if count > 200:
            hlLF.flush()
            count = 0
    hlLF.close()

def generateHistFromLab(imgNames):
    labF = open(rootPath + '\\' + "lab_hist_unify.txt", "a")
    count = 0
    process = 0
    for imgName in imgNames:
        print str(process) + '/' + str(len(imgNames))
        process += 1
        point = getPoint(imgName)
        img = cv2.imread(imgName)
        if img is None:
            continue
        img = adjustSize(img)
        if img is None:
            continue
        b, g, r = cv2.split(img)
        if b.tolist() == g.tolist() and g.tolist() == r.tolist():
            print 'gray image'
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        height = img.shape[0]
        width = img.shape[1]
        hist = getImageHist(img)
        hist = map(lambda x : round(float(x)/(height*width), 3), hist)
        if len(hist) == 0:
            print 'unexpected error!'
            continue
        write(process, labF, hist, point)
        count += 1
        if count > 200:
            labF.flush()
            count = 0
    labF.close()

def generateHistFromXyz(imgNames):
    labF = open(rootPath + '\\' + "xyz_hist_unify.txt", "a")
    count = 0
    process = 0
    for imgName in imgNames:
        print str(process) + '/' + str(len(imgNames))
        process += 1
        point = getPoint(imgName)
        img = cv2.imread(imgName)
        if img is None:
            continue
        img = adjustSize(img)
        if img is None:
            continue
        b, g, r = cv2.split(img)
        if b.tolist() == g.tolist() and g.tolist() == r.tolist():
            print 'gray image'
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
        height = img.shape[0]
        width = img.shape[1]
        hist = getImageHist(img)
        hist = map(lambda x : round(float(x)/(height*width), 3), hist)
        if len(hist) == 0:
            print 'unexpected error!'
            continue
        write(process, labF, hist, point)
        count += 1
        if count > 200:
            labF.flush()
            count = 0
    labF.close()

def calcMoment(channel, mean, level):
    N = len(channel) * len(channel[0])
    subChannel = channel - mean
    expChannel = subChannel ** level
    return round((expChannel.sum() / N) ** (1.0 / level), 6)

def generateColorMoment(imgNames):
    cmF = open(rootPath + '\\' + "rgb_color_moment.txt", "a")
    count = 0
    process = 0
    for imgName in imgNames:
        print str(process) + '/' + str(len(imgNames))
        process += 1
        point = getPoint(imgName)
        img = cv2.imread(imgName)
        if img is None:
            continue
        img = adjustSize(img)
        if img is None:
            continue
        b, g, r = cv2.split(img)
        if b.tolist() == g.tolist() and g.tolist() == r.tolist():
            print 'gray image'
            continue
        N = len(b) * len(b[0])
        bMean = b.sum() / N
        gMean = g.sum() / N
        rMean = r.sum() / N
        moment = [bMean, calcMoment(b, bMean, 2), calcMoment(b, bMean, 3),
        gMean, calcMoment(g, gMean, 2), calcMoment(g, gMean, 3),
        rMean, calcMoment(r, rMean, 2), calcMoment(r, rMean, 3)]
        write(process, cmF, moment, point)
        count += 1
        if count > 200:
            cmF.flush()
            count = 0
    cmF.close()

def generateColorMoments(imgNames):
    cmF = open(rootPath + '\\' + "rgb_color_moments.txt", "a")
    count = 0
    process = 0
    for imgName in imgNames:
        print str(process) + '/' + str(len(imgNames))
        process += 1
        point = getPoint(imgName)
        img = cv2.imread(imgName)
        if img is None:
            continue
        img = adjustSize(img)
        if img is None:
            continue
        b, g, r = cv2.split(img)
        if b.tolist() == g.tolist() and g.tolist() == r.tolist():
            print 'gray image'
            continue
        bs, gs, rs =[], [], []
        hstep = b.shape[0] / 8
        wstep = b.shape[1] / 8
        for i in range(8):
            for j in range(8):
                bs.append(b[i*hstep:(i+1)*hstep, j*wstep:(j+1)*wstep])
                gs.append(g[i*hstep:(i+1)*hstep, j*wstep:(j+1)*wstep])
                rs.append(r[i*hstep:(i+1)*hstep, j*wstep:(j+1)*wstep])
        N = len(b) * len(b[0])
        moments = []
        for i in range(8):
            bMean = bs[i].sum() / N
            gMean = gs[i].sum() / N
            rMean = rs[i].sum() / N
            moment = [bMean, calcMoment(bs[i], bMean, 2), calcMoment(bs[i], bMean, 3),
            gMean, calcMoment(gs[i], gMean, 2), calcMoment(gs[i], gMean, 3),
            rMean, calcMoment(rs[i], rMean, 2), calcMoment(rs[i], rMean, 3)]
            moments += moment
        write(process, cmF, moments, point)
        count += 1
        if count > 200:
            cmF.flush()
            count = 0
    cmF.close()

def generateArtHistogram(filename):
    afF = open(rootPath + '\\' + "hsv_art_histogram.txt", "a")
    lines = open(filename).readlines()
    count = 0
    process = 0
    for line in lines:
        print str(process) + '/' + str(len(imgNames))
        process += 1
        data = line.split('\t')[1].split(' ')
        point = data[-1]
        histogram = map(float, data[:-1])
        cpdata = histogram[:]
        value = []
        indexs = []
        maxValue, index = 0, - 1
        while True:
            for i in range(180):
                if histogram[i] > maxValue:
                    maxValue = histogram[i]
                    index = i
            if len(value) == 0 or maxValue > 0.3 * value[0]:
                value.append(maxValue)
                indexs.append(index)
                maxValue, index = 0, - 1
                if len(indexs) == 3:
                    break
                for j in range(i - 10, i + 10):
                    histogram[(j + 180) % 180] = 0
            else:
                break
        feature = [0] * 60
        for i in range(len(indexs)):
            down, up = indexs[i] - 10, indexs[i] + 10
            seg = []
            if down >= 0:
                seg = cpdata[down : up]
            else:
                seg = cpdata[down:] + cpdata[:up]
            feature[i * 20 : (i+1) * 20] = seg[:20]
        write(process, afF, feature, point)
        count += 1
        if count > 200:
            afF.flush()
            count = 0
    afF.close()

def generateArtFeature(filename):
    afF = open(rootPath + '\\' + "hsv_art_feature_cd.txt", "a")
    lines = open(filename).readlines()
    count = 0
    process = 0
    for line in lines:
        print str(process) + '/' + str(len(imgNames))
        process += 1
        data = line.split('\t')[1].split(' ')
        point = data[-1]
        histogram = map(float, data[:-1])
        value = []
        indexs = []
        maxValue, index = 0, -1
        while True:
            for i in range(180):
                if histogram[i] > maxValue:
                    maxValue = histogram[i]
                    index = i
            if len(value) == 0 or maxValue > 0.3 * value[0]:
                value.append(maxValue)
                indexs.append(index)
                maxValue, index = 0, -1
                if len(indexs) == 3:
                    break
                for j in range(i - 10, i + 10):
                    histogram[(j + 180) % 180] = 0
            else:
                break
        feature = [0, 0, 0, 0, 0, 0]
        for i in range(len(indexs)):
            if len(indexs) == 1:
                feature[0] = 1
            for j in range(i + 1, len(indexs)):
                if isNeighborColor(indexs[i], indexs[j]):
                    feature[1] = 1
                if isLikeColor(indexs[i], indexs[j]):
                    feature[2] = 1
                if isCompareColor(indexs[i], indexs[j]):
                    feature[3] = 1
                if isCompleteColor(indexs[i], indexs[j]):
                    feature[4] = 1
        if sum(value) > 0.8:
            feature[5] = 1
        write(process, afF, feature, point)
        count += 1
        if count > 200:
            afF.flush()
            count = 0
    afF.close()

def colorDistance(h1, h2):
    dis = abs(h1 - h2)
    if dis > 180:
        dis = 360 - dis
    return dis

def isNeighborColor(h1, h2):
    if abs(colorDistance(h1, h2) - 22.5) <= 2:
        return True
    return False

def calNeighborColor(h1, h2):
    return abs(colorDistance(h1, h2) - 22.5) / 180.0

def isLikeColor(h1, h2):
    if abs(colorDistance(h1, h2) - 45) <= 4:
        return True
    return False

def calLikeColor(h1, h2):
    return abs(colorDistance(h1, h2) - 45) / 180.0

def isCompareColor(h1, h2):
    if colorDistance(h1, h2) >= 68 and colorDistance(h1, h2) <= 114:
        return True
    return False

def calCompareColor(h1, h2):
    d = min(abs(colorDistance(h1, h2) - 68), abs(colorDistance(h1, h2) - 114))
    return d / 180.0

def isCompleteColor(h1, h2):
    if abs(colorDistance(h1, h2) - 180) <= 16:
        return True
    return False

def calCompleteColor(h1, h2):
    return abs(colorDistance(h1, h2) - 180) / 180.0

def generateArtFeature2(filename):
    afF = open(rootPath + '\\' + "hsv_art_feature_cd_percent.txt", "a")
    lines = open(filename).readlines()
    count = 0
    process = 0
    for line in lines:
        print str(process) + '/' + str(len(imgNames))
        process += 1
        data = line.split('\t')[1].split(' ')
        point = data[-1]
        histogram = map(float, data[:-1])
        value = []
        indexs = []
        maxValue, index = 0, -1
        while True:
            for i in range(180):
                if histogram[i] > maxValue:
                    maxValue = histogram[i]
                    index = i
            if len(value) == 0 or maxValue > 0.3 * value[0]:
                value.append(maxValue)
                indexs.append(index)
                maxValue, index = 0, -1
                if len(indexs) == 3:
                    break
                for j in range(i - 10, i + 10):
                    histogram[(j + 180) % 180] = 0
            else:
                break
        feature = [0, 0, 0, 0, 0, 0]
        dist = [1, 1, 1, 1, 1]
        for i in range(len(indexs)):
            if len(indexs) == 1:
                feature[0] = 1
                dist[0] = 0
            for j in range(i + 1, len(indexs)):
                if isNeighborColor(indexs[i], indexs[j]):
                    feature[1] = 1
                if isLikeColor(indexs[i], indexs[j]):
                    feature[2] = 1
                if isCompareColor(indexs[i], indexs[j]):
                    feature[3] = 1
                if isCompleteColor(indexs[i], indexs[j]):
                    feature[4] = 1
                if calNeighborColor(indexs[i], indexs[j]) < dist[1]:
                    dist[1] = calNeighborColor(indexs[i], indexs[j])
                if calLikeColor(indexs[i], indexs[j]) < dist[2]:
                    dist[2] = calLikeColor(indexs[i], indexs[j])
                if calCompareColor(indexs[i], indexs[j]) < dist[3]:
                    dist[3] = calCompareColor(indexs[i], indexs[j])
                if calCompleteColor(indexs[i], indexs[j]) < dist[4]:
                    dist[4] = calCompleteColor(indexs[i], indexs[j])
        if sum(value) > 0.8:
            feature[5] = 1
        write(process, afF, feature + dist, point)
        count += 1
        if count > 200:
            afF.flush()
            count = 0
    afF.close()

if __name__ == '__main__':
    loadImgs('D:\code\commicDownload')
    print 'have ' + repr(len(imgs)) + ' images'
    bottom, top = slip(imgs)
    allTrain = bottom + top
    imgNames = [x[1]['filename'] for x in allTrain]
    random.shuffle(imgNames)
    #generateFromRGB(imgNames)
    ##generateFromHSV(imgNames)
    #generateHistFromHSV(imgNames)
    #generateHistFromLab(imgNames)
    #generateHistFromXyz(imgNames)
    #generateColorMoment(imgNames)
    #generateArtFeature2(rootPath + '\\' + "hsv_hist_new_unify.txt")
    #generateArtHistogram(rootPath + '\\' + "hsv_hist_new_unify.txt")
    #generateHistFromHLS(imgNames)
    #generateHistFromHLL(imgNames)
    #generateDifHistFromHSV(imgNames)
    #generateFromDifImg(imgNames)
    #generateSumImg(imgNames)
    generateColorMoments(imgNames)