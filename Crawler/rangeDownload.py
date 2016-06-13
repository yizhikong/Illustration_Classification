# -*- coding: utf-8 -*-
import re
import urllib2
import urllib
import cookielib
import threading
import sys
import time

def extractOne(opener, detailUrl):
    illust_id = detailUrl.split('=')[-1]
    request = urllib2.Request(detailUrl)
    html = opener.open(request).read()
    scorePat = re.compile('<dd class="view-count">(\d+)</dd>.*?<dd class="rated-count">(\d+)</dd>.*?<dd class="score-count">(\d+)</dd>')
    imgUrlPat = re.compile('data-src="(.*?)" class="original-image"')
    r18Pat = re.compile('<li class="r-18">R-18</li>')
    msg = ''
    res = r18Pat.search(html, re.MULTILINE)
    if res:
        print 'R18 found, ignore'
        return
    res = scorePat.search(html, re.MULTILINE)
    if res:
        msg = '%s_%s_%s_%s' % (illust_id, res.group(1), res.group(2), res.group(3))
    else:
        return
    filename = msg + '.jpg'
    res = imgUrlPat.search(html, re.MULTILINE)
    if res:
        imgUrl = res.group(1)
    else:
        print filename + ' not found image url'
        return
    print imgUrl
    request = urllib2.Request(imgUrl)
    img = opener.open(request).read()
    f = open(filename, "wb")
    f.write(img)
    f.close()
    
def getPixivOpener():
    cj = cookielib.CookieJar()
    opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
    loginUrl = 'https://www.secure.pixiv.net/login.php'
    indexUrl = 'http://www.pixiv.net/'
    data = urllib.urlencode({'mode':'login',
                             'return_to':'/',
                             'pixiv_id':'yzkk',
                             'pass':'mypassword',
                             'skip':'1'})
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

if __name__ == '__main__':
    opener = getPixivOpener()
    # target : 53160000 - 53170000
    # 10/26/2015, 53160000 - 53165000
    # 10/27/2015, 53165000 - 53170000
    # 10/30/2015, 53170000 - 53175000
    # 10/31/2015, 53175000 - 53180000
    # 11/02/2015, 53180000 - 53185000
    # 11/03/2015. 53185000 - 53190000 reduce r18
    # 11/05/2015, 53190000 - 53195000 reduce r18， stop at 53192218
    # 11/07/2015, 53195000 - 53200000
    # file system full, should move to my computer
    detailUrl = 'http://www.pixiv.net/member_illust.php?mode=medium&illust_id='
    for i in range(53550000, 53560000):
        try:
            extractOne(opener, detailUrl + str(i))
        except:
            print '403 occur'
            time.sleep(30)
