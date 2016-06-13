sourceFile = r'hsvdif_hist_new_unify.txt'
targetFile = r'hsvdif_hist_new_unify_45C.txt'

def quantize(level, nums):
    nums = map(lambda x : float(x), nums)
    point = int(nums[-1])
    nums = nums[:-1]
    step = 256 / level
    i = 0
    result = []
    while i < len(nums):
        result.append(sum(nums[i:i+step]))
        i += step
    result.append(point)
    return result

def hsvQuantize(nums, hstep, svstep):
    nums = map(lambda x : float(x), nums)
    point = int(nums[-1])
    nums = nums[:-1]
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
    result.append(point)
    return result

def hsvQuantize2(nums, hstep, svstep, end):
    source = map(lambda x : float(x), nums)
    source[-1] = int(source[-1])
    nums = source[:end]
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
    source[:end] = result[:]
    return source

def colorBaseTransform(nums, down, up, offset = 0):
    nums = map(lambda x : float(x), nums)
    source = nums[down:up]
    colorNum = up - down
    merge = 360 / colorNum
    angles = [30, 60, 90, 270, 300, 330]
    for i in range(len(angles)):
        angles[i] /= merge
    d = len(angles) * (offset + 1)
    for i in range(down, up):
        for angle in angles:
            #print '---------%d----------%d' % (i, angle)
            st = i + angle - offset
            end = i + angle + 1 + offset
            if end < 0:
                bins = source[st+colorNum:end+colorNum]
                #print '%d to %d' % (st+colorNum, end+colorNum)
            elif st < 0 and end >= 0:
                bins = source[st+colorNum:] + source[:colorNum]
                #print '%d to : and : to %d' % (st+colorNum, colorNum)
            elif st >= 0 and end <= colorNum:
                bins = source[st:end]
                #print '%d to %d' % (st, end)
            elif st >= colorNum:
                bins = source[st-colorNum:end-colorNum]
                #print '%d to %d' % (st-colorNum, end-colorNum)
            elif st < colorNum and end > colorNum:
                bins = source[st:] + source[:end-colorNum]
                #print '%d to : and : to %d' % (st, end-colorNum)
            else:
                print 'error'
            nums[i] += sum(bins) / 2
    nums[-1] = int(nums[-1])
    return nums

def smooth(nums, down, up, size):
    nums = map(lambda x : float(x), nums)
    segValue = nums[down:up]
    source = segValue[-1-size/2:-1] + segValue + segValue[0:size/2]
    target = []
    for i in range(up - down):
        target.append(sum(source[i : i + size]) / size)
    if len(target) != up - down:
        print 'error'
    nums[down:up] = target[:]
    nums[-1] = int(nums[-1])
    return nums

def sortSmooth(nums, down, up, size):
    nums = map(lambda x : float(x), nums)
    seg = nums[down:up]
    for i in range(len(seg)):
        seg[i] = (seg[i], i)
    seg = sorted(seg, key = lambda x : x[0])
    segValue = map(lambda x : x[0], seg)
    source = segValue[-1-size/2:-1] + segValue + segValue[0:size/2]
    target = []
    for i in range(up - down):
        target.append((sum(source[i : i + size]) / size, seg[i][1]))
    target = sorted(target, key = lambda x : x[1])
    target = map(lambda x : x[0], target)
    if len(target) != up - down:
        print 'error'
    nums[down:up] = target[:]
    nums[-1] = int(nums[-1])
    return nums

def loadData():
    f = open(sourceFile)
    lines = f.readlines()
    f.close()
    data = []
    for i in range(len(lines)):
        line = lines[i][:]
        lines[i] = ''
        pos = line.find(' ')
        if pos < 0:
            continue
        line = line[pos+1 :].strip()
        spLine = line.split(' ')
        data.append(spLine)
    return data

def write(index, fop, hist):
    fop.write('index:' + str(index) + '\t')
    for i in range(len(hist) - 1):
        fop.write(str(hist[i]) + ' ')
    fop.write(str(hist[-1]) + '\n')

def mergeChannel(cFiles, outputFile):
    cdata = []
    for cfile in cFiles:
        cdata.append(open(cfile).readlines())
    f = open(outputFile, 'w')
    count = 0
    for i in range(len(cdata[0])):
        values = []
        marks = []
        for j in range(len(cdata)):
            line = cdata[j][i]
            values.append((line.split('\t')[1]).split(' ')[:-1])
            marks.append(line.split(' ')[-1])
        flag = True
        for j in range(len(marks)):
            if marks[0] != marks[j]:
                flag = False
        if flag:
            hist = values[0]
            for j in range(1, len(values)):
                hist += values[j]
            hist.append(int(marks[0]))
            write(i, f, hist)
        else:
            print 'error'
        if count % 10 == 0:
            f.flush()
            print count
        count += 1
    f.close()

if __name__ == '__main__':
    #mergeChannel(['hsv_hist_new_unify_merge2.txt', 'gray_clear_new_unify_64l.txt'], 'hsv_gray_hist_unify.txt')
    #a = input()
    data = loadData()
    f = open(targetFile, 'w')
    count = 0
    for i in range(len(data)):
        #write(i, f, quantize(64, data[i]))
        #h = data[i][180:180+256]
        #h.append(data[i][-1])
        #write(i, f, h)
        #write(i, f, colorBaseTransform(smooth(data[i], 0, 90, 5), 0, 90, 0))
        #write(i, f, hsvQuantize(data[i], 4, 4))
        write(i, f, hsvQuantize2(data[i], 4, 8, 691))
        #write(i, f, colorBaseTransform(data[i], 0, 90, 1))
        if count % 10 == 0:
            f.flush()
            print count
        count += 1
    f.close()
