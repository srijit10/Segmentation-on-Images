import numpy as np
import cv2
import matplotlib.pyplot as plt

#def bht(img):
img=cv2.imread("188_1_1_back.jpg",cv2.IMREAD_GRAYSCALE)
cols,rows= img.shape
occurances = np.zeros(256, dtype=np.int)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img = clahe.apply(img)
for y in xrange(0, cols):
    for x in xrange(0, rows):
        occurances[img[y][x]] = occurances[img[y][x]] + 1

X =np.zeros(shape=(len(occurances),2))
for i in xrange(0,256):
    if(occurances[i]>0):
        X[i][0]=i
        X[i][1]=occurances[i]

plt.scatter(X[:, 0], X[:, 1])
plt.show()

# Apply double thresholding to remove noise
# Threshold1=40 , Threshold2=100
#for spider threshold1=70 ,threshold2=170

for y in xrange(0,cols):
    for x in xrange(0,rows):
        if (img[y][x]<70 or img[y][x]>170):
            img[y][x]=0

'''
occurances = np.zeros(256, dtype=np.int)

for y in xrange(0, cols):
    for x in xrange(0, rows):
        occurances[img[y][x]] = occurances[img[y][x]] + 1

X =np.zeros(shape=(len(occurances),2))
for i in xrange(0,256):
    if(occurances[i]>0):
        X[i][0]=i
        X[i][1]=occurances[i]

plt.scatter(X[:, 0], X[:, 1])
plt.show()
'''
for i in xrange(0,256):
    if(occurances[i]>0):
        start=i #start is the first graylevel intensity present in the image
        break


for i in xrange(255,0,-1):
    if(occurances[i]>0):
        end=i #end is the last graylevel intensity present in the image
        break

print start,end
mid=int((start+end)/2.0)


wl=0
for i in xrange(start,mid+1):
    wl+=occurances[i]
wr=0
for i in xrange(mid+1,end+1):
    wr+=occurances[i]

#---------Given Algorithm---------#

while(start<=end):
    #right side is heavier
    if(wr>wl):
        wr-=occurances[end]
        end=end-1
        if (int((start+end)/2) <mid):
            wr+=occurances[mid]
            wl-=occurances[mid]
            mid=mid-1
    #left side is heavier
    elif (wl>=wr):
        wl-= occurances[start]
        start=start+1
        if (int((start + end) / 2) >= mid):
            wl+=occurances[mid + 1]
            wr-=occurances[mid + 1]
            mid=mid+1

threshold=mid
print threshold

for y in xrange(0, cols):
    for x in xrange(0, rows):
        if (img[y][x] > threshold):
            img[y][x] = 0
        else:
            img[y][x] = 255
cv2.imshow("bht",img)
cv2.waitKey(0)