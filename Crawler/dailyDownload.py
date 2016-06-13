# -*- coding: utf-8 -*-
import re
import urllib2
import urllib
import cookielib
import threading
import sys
import datetime
import threading
import os
import time

'''
each rank page has 50 illust
first generate many rank pages
then get illusts from each rank page 
'''

# avoid duplicate
illustDict = {}
duplicateCount = 0
filtCount = 0
savepath = 'D:\code\commicDownload\\rankDown'
pixiv_id = 'yzkk'
password = 'mypassword'
mutex = None

'''[function for multiple thread]'''
def extractRankPage(rankPage):
    # get illusts' url from the rank page
    urlList = generateIllustUrlList(rankPage)
    for url in urlList:
        # extract each illust
        try:
            extractIllust(url)
            time.sleep(1)
        except:
            print url[2] + ' 404'
   
def extractIllust(urlTuple):
    """Extract one tuple(rank, illust_id, imgUrl), get illust's information and download it.

    urlTuple 

    """
    rank, illust_id, imgUrl = urlTuple
    if mutex:
        mutex.acquire(1)
    if illust_id not in illustDict:
        illustDict[illust_id] = True
        mutex.release()
    else:
        global duplicateCount
        print '[[' + illust_id + ' has downloaded' + ']]'
        duplicateCount += 1
        print '[[' + 'Current duplicate count is ' + str(duplicateCount) + ']]'
        mutex.release()
        return
    detailUrl = 'http://www.pixiv.net/member_illust.php?mode=medium&illust_id=' + illust_id
    request = urllib2.Request(detailUrl)
    html = opener.open(request).read()
    scorePat = re.compile('<dd class="view-count">(\d+)</dd>.*?<dd class="rated-count">(\d+)</dd>.*?<dd class="score-count">(\d+)</dd>')
    imgUrlPat = re.compile('data-src="(.*?)" class="original-image"')
    
    # avoid these set(always advertisement)
    multiplePat = re.compile('一次性投稿多张作品')
    res = multiplePat.search(html, re.MULTILINE)
    if res:
        if mutex:
            mutex.acquire(1)
        global filtCount
        print '[[' + illust_id + ' has multiple pictures' + ']]'
        filtCount += 1
        print '[[' + 'Current filt count is ' + str(filtCount) + ']]'
        mutex.release()
        return

    # get illust's message use regular expression
    msg = ''
    res = scorePat.search(html, re.MULTILINE)
    if res:
        msg = '%s_%s_%s_%s' % (res.group(1), res.group(2), res.group(3), illust_id)
    filename = str(rank) + '_' + msg + '.jpg'
    res = imgUrlPat.search(html, re.MULTILINE)
    if res:
        imgUrl = res.group(1)

    # download the image
    request = urllib2.Request(imgUrl)
    img = opener.open(request).read()
    f = open(savepath + '\\' + filename, "wb")
    f.write(img)
    f.close()
    print filename
    #urllib.urlretrieve(imgUrl, filename)

'''[login pixiv, get the opener with cookie]'''
def getPixivOpener():
    cj = cookielib.CookieJar()
    opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
    loginUrl = 'https://www.secure.pixiv.net/login.php'
    indexUrl = 'http://www.pixiv.net/'
    data = urllib.urlencode({'mode' : 'login',
                             'return_to' : '/',
                             'pixiv_id' : pixiv_id,
                             'pass' : password,
                             'skip' : '1'})
    opener.addheaders = [
        ('User-Agent', 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko'),
        ('Referer', 'http://www.pixiv.net'),
        ('Accept-Language', 'en-US,en'),
        ('Content-Type', 'application/x-www-form-urlencoded'),
        ('Host', 'www.secure.pixiv.net'),
        ('Accept-Encoding', 'deflate, peerdist'),
        ('Connection', 'Keep-Alive'),
        ('X-P2P-PeerDist', 'Version=1.1'),
        ('X-P2P-PeerDistEx', 'MinContentInformation=1.0, MaxContentInformation=1.0')
        ]
    home = opener.open(loginUrl, data)
    return opener

# opener with cookie
opener = getPixivOpener()

'''[generate rank page urls, the url can be constructed by date and p]'''
def generateRankPageUrlList(beginDate, endDate):
    inc = datetime.timedelta(days = 1)
    timeFormat = '%Y%m%d'
    urlList = []
    # get these base url by analyzing the get request in website
    baseUrl = 'http://www.pixiv.net/ranking.php?mode=daily&content=illust&date=%s&p=%s'
    while beginDate <= endDate:
        for p in range(1, 11):
            dateStr = beginDate.strftime(timeFormat)
            urlList.append(baseUrl % (dateStr, str(p)))
        beginDate += inc
    return urlList

'''[generate illust urls by using regular expression in a rank page]'''
def generateIllustUrlList(rankPageUrl):
    request = urllib2.Request(rankPageUrl)
    html = opener.open(request).read()
    # use regular expression to extract the image url from the rank page
    detailPat = re.compile('data-rank-text="#(\d+)".*?illust_id=(\d+)')
    imgPat = re.compile('<div class="ranking-image-item">.*?data-filter="lazy-image".*?data-src="(.*?)"></div></a></div><h2><a href="member_illust')
    detailUrls = detailPat.findall(html, re.MULTILINE)
    imgUrls = imgPat.findall(html, re.MULTILINE)
    # exchange to list
    page = int(rankPageUrl[rankPageUrl.find('p=')+2 : ])
    urlList = []
    for i in range(len(detailUrls)):
        urlList.append((i + (page - 1) * 50, detailUrls[i][1], imgUrls[i]))
    return urlList

def reprocess():
    # load local illust id
    if os.path.exists(savepath):
        filenames = os.listdir(savepath)
        illustIds = [filename.split('.')[0].split('_')[-1] for filename in filenames]
        for illust in illustIds:
            if illust not in illustDict:
                illustDict[illust] = True
        print 'Load ' + str(len(illustDict)) + ' illust ids from local'
        return len(filenames)
    else:
        os.mkdir(savepath)
        return 0

if __name__ == '__main__':
    beforeCount = reprocess()
    # get the list of rank_pages' url by date
    beginDate = datetime.datetime(2015, 1, 1)
    endDate = datetime.datetime(2015, 1, 3)
    dec = datetime.timedelta(days = -4)
    mutex = threading.Lock()
    while beginDate >= datetime.datetime(2015, 1, 1):
        f = open('D:\dateRecord.txt', 'a')
        f.write('begin : ' + beginDate.strftime('%Y%m%d') + '\n')
        f.write('end : ' + endDate.strftime('%Y%m%d') + '\n')
        f.close()
        rankPageUrlList = generateRankPageUrlList(beginDate, endDate)
        threads = []
        for pageUrl in rankPageUrlList:
            print 'Extracting rank page ' + pageUrl
            t = threading.Thread(target = extractRankPage, args = (pageUrl,))
            t.start()
            threads.append(t)
        for th in threads:
            th.join(300)
        beginDate += dec
        endDate += dec
    print '----------------------------'
    filenames = os.listdir(savepath)
    afterCount = len(filenames)
    print 'before : ' + str(beforeCount)
    print 'after : ' + str(afterCount)
    print 'filt : ' + str(filtCount)
    print 'multiple : ' + str(duplicateCount)
    print 'sum : ' + str(filtCount + duplicateCount + afterCount - beforeCount)
    print 'Finish'
