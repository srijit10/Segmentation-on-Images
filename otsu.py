import cv2
import numpy as np


def otsu(img):
    cols,rows =img.shape
    # Apply double thresholding to remove noise
    # Threshold1=40 , Threshold2=100
    for y in xrange(0,cols):
        for x in xrange(0,rows):
            if (img[y][x]<40 or img[y][x]>100):
                img[y][x]=0
    #cv2.imshow('doublethreshold',img)
    # Otsu Thresholding
    maxgraylevel=-1
    occurances=np.zeros(256, dtype=np.int)

    for y in xrange(0,cols):
        for x in xrange(0,rows):
            if (maxgraylevel<img[y][x]):
                maxgraylevel=img[y][x]
            occurances[img[y][x]]=occurances[img[y][x]]+1

    

    totpix=cols*rows
    min_withinclass=99999999
    # xrange takes values from 0 to maxgraylevel hence it is written maxgraylevel+1
    for k in xrange(0,maxgraylevel+1):
        w1=0
        w2=0
        m1=0
        m2=0
        v1=0
        v2=0
        #--------Class C1----------
        for i in xrange(0,k+1):
            w1+=occurances[i]
            m1+=i*occurances[i]
        m1=m1/w1
        for i in xrange(0,k+1):
            v1+=(i-m1)*(i-m1)*occurances[i]
        v1=v1/w1
        w1 =w1/totpix
        #--------Class C2-----------
        for i in xrange(k,maxgraylevel+1):
            w2+= occurances[i]
            m2+=i*occurances[i]
        m2=m2/w2;
        for i in xrange(k,maxgraylevel+1):
            v2+=(i-m2)*(i-m2)*occurances[i]
        v2=v2/w2
        w2=w2/totpix
        withinclass=(w1 * v1)+(w2 * v2)
        if(min_withinclass> withinclass):
            min_withinclass = withinclass;
            threshold = k;
    threshold=10
    for y in xrange(0,cols):
        for x in xrange(0,rows):
            if(img[y][x]>threshold):
                img[y][x]=0
            else:
                img[y][x]=255

    #cv2.imshow('otsuthreshold',img)
    #cv2.waitKey(0)
    return img

#img=cv2.imread("200px-Lovely_spider.jpeg",cv2.IMREAD_GRAYSCALE)
#otsu_img=otsu(img)
