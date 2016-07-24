import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from sklearn import svm
from matplotlib import style
from histogramthresholding import histogramthresholding
from otsu import otsu
import random


testarr=['188_1_1_back.jpg','188_1_2_back.jpg','188_2_1_back.jpg','190_4_1_back.jpg','190_5_1_back.jpg',
          '190_7_1_back.jpg','188_3_1_back.jpg','188_7_1_back.jpg','188_3_2_back.jpg','188_4_1_back.jpg'
          ]

#mritop =['188_1_1_top.jpg','188_1_2_top','188_2_1_top.jpg','188_3_1_top.jpg','188_3_2_top.jpg',
#         '188_4_1_top.jpg','188_5_1_top.jpg','188_6_1_top.jpg']


arr= [    '188_9_1_back.jpg','189_1_1_back.jpg','189_1_2_back.jpg','189_2_1_back.jpg','189_3_1_back.jpg',
          '189_3_2_back.jpg','189_4_1_back.jpg','189_5_1_back.jpg','189_6_1_back.jpg','189_6_2_back.jpg',
          '189_7_1_back.jpg','189_9_1_back.jpg','189_10_1_back.jpg','190_1_1_back.jpg','190_1_2_back.jpg',
          '190_2_1_back.jpg','190_3_1_back.jpg','190_3_2_back.jpg','188_5_1_back.jpg','188_5_1_back.jpg',
          '188_6_2_back.jpg']

path=['/home/srijit/PycharmProjects/Glcm-svm/mriback/']

diss=[]
corr=[]
classes=[]
for j in xrange(0,len(arr)):
    whitematter = []
    greymatter = []
    img=cv2.imread(path[0]+arr[j],cv2.IMREAD_GRAYSCALE)
    # create a CLAHE object (Arguments are optional).  To make the image more distinct
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    #cv2.imshow('original', img)

    #Cropping the image
    cropped = img[36:187, 38:253]

    #cv2.imshow("cropped", cropped)
    #cv2.waitKey(0)

    otsu_img=otsu(cropped)
    #thres_img = histogramthresholding(cropped)  # Balanced histogram thresholding
    cols,rows=otsu_img.shape
    # get the whitematter and graymatter patches using otsu thresholding for the original image
    for y in xrange(0,cols,10):
        for x in xrange(0,rows,10):
            if(otsu_img[y][x]==255):
                whitematter.append(img[y:y+10,x:x+10])
            if(otsu_img[y][x]==0):
                greymatter.append(img[y:y+10,x:x+10])
    #Calculate glcm parameters for the white matter
    for i in xrange(0,5):
        glcm = greycomatrix(whitematter[i], [1], [0], 256, symmetric=True, normed=True)
        diss.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        corr.append(greycoprops(glcm, 'contrast')[0, 0])
        classes.append(0)

    # Calculate glcm parameters for the grey matter
    for i in xrange(0,5):
        glcm = greycomatrix(greymatter[i], [1], [0], 256, symmetric=True, normed=True)
        diss.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        corr.append(greycoprops(glcm, 'contrast')[0, 0])
        classes.append(1)


clf = svm.SVC(kernel='linear', C=1.0)
# train the svm using the dissimilarity and correlation parameters for the white and gray matter
X =np.zeros(shape=(len(diss),2))
for i in xrange(0, len(corr)):
    X[i][0] = diss[i]
    X[i][1] = corr[i]


# for whitematter Class = 0 and for greymatter Class = 1

clf.fit(X, classes)

w = clf.coef_[0]
a = -w[0] / w[1]

# 50 values of x
xx = np.linspace(min(X[:,0]),max(X[:,0]))
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c=classes)
plt.ylabel('contrast')
plt.xlabel('dissimilarity')
plt.show()
plt.legend()

'''
# test the svm
for i in range(0,len(testarr)):
    w=[]
    g=[]
    image=cv2.imread(path[0] + testarr[i], cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    cropped = image[36:187, 38:253]
    otsu_img = otsu(cropped)

    #thres_img=histogramthresholding(cropped)#Balanced histogram thresholding

    cols, rows = otsu_img.shape
    # get the whitematter and graymatter using otsu thresholding for the original image
    for y in xrange(0, cols):
        for x in xrange(0, rows):
            if (otsu_img[y][x] == 255):
                w.append([y, x])
            if (otsu_img[y][x] == 0):
                g.append([y, x])
    glcm = greycomatrix(w, [1], [0], 256, symmetric=True, normed=True)
    a=(greycoprops(glcm, 'dissimilarity')[0, 0])
    b=(greycoprops(glcm, 'contrast')[0, 0])
    print i+1,a,b,clf.predict([[a,b]])
    glcm = greycomatrix(g, [1], [0], 256, symmetric=True, normed=True)
    a = (greycoprops(glcm, 'dissimilarity')[0, 0])
    b = (greycoprops(glcm, 'contrast')[0, 0])
    print i+1,a, b, clf.predict([[a, b]])
'''
PATCH_SIZE =2
for i in xrange(0,len(testarr)):
    w=[]
    image=cv2.imread(path[0] + testarr[i], cv2.IMREAD_GRAYSCALE)
    cropped = image[36:187, 38:253]
    cols,rows=cropped.shape
    (y,x)=random.randint(0, cols),random.randint(0, rows)
    fig = plt.figure(figsize=(4, 4))
    # display original image with locations of patches
    ax = fig.add_subplot(1, 1, 1)
    # show the image
    ax.imshow(cropped, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
    ax.set_xlabel('Original Image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')
    plt.show()
    w.append(image[y:y+PATCH_SIZE,x:x+PATCH_SIZE])
    glcm = greycomatrix(w[0], [1], [0], 256, symmetric=True, normed=True)
    a = (greycoprops(glcm, 'dissimilarity')[0, 0])
    b = (greycoprops(glcm, 'contrast')[0, 0])
    if(clf.predict([[a,b]])==0):
        print i+1,a,b,"whitematter portion"
    else:
        print i+1,a,b,"grey matter portion"









'''for i in range(0,1):
    image = cv2.imread(path[0] + testarr[i], cv2.IMREAD_GRAYSCALE)
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    # Apply glcm for the whole image
    glcm = greycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
    a.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    b.append(greycoprops(glcm, 'correlation')[0, 0])
    p = a[0]
    q = b[0]
    print p
    print q
    print(clf.predict([p,q]))
'''
'''
for i in xrange(0,len(whitematter)):
    plt.scatter(whitematter[i][0],whitematter[i][1],c=1)
plt.show()
'''
'''
bs=[]
cs=[]
y=[0,1,0,1,0,1,0,1,0,1]
for i in xrange(0,len(arr)):
    # Load an image in grayscale
    img=cv2.imread(path[0]+arr[i],cv2.IMREAD_GRAYSCALE)
    # create a CLAHE object (Arguments are optional).
    clahe= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl=clahe.apply(img)

    glcm = greycomatrix(cl,[1],[0], 256, symmetric=True, normed=True)
    bs.append(greycoprops(glcm, 'dissimilarity')[0,0])
    cs.append(greycoprops(glcm, 'correlation')[0,0])

clf = svm.SVC(kernel='linear', C = 1.0)

X =np.zeros((len(bs),len(bs)))
for i in xrange(0,len(bs)):
    X[i][0]=bs[i]
    X[i][1]=cs[i]

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.show()

clf.fit(X,y)

a=[]
b=[]
#test the images to which classes it lies
#for i in xrange(0,len(testarr)):
image=cv2.imread(path[0]+testarr[0],cv2.IMREAD_GRAYSCALE)
# create a CLAHE object (Arguments are optional).
clahe= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl=clahe.apply(image)
glcm = greycomatrix(cl,[1],[0], 256, symmetric=True, normed=True)
a.append(greycoprops(glcm, 'dissimilarity')[0,0])
b.append(greycoprops(glcm, 'correlation')[0,0])
p=a[0]
q=b[0]
print p
print q
#print(clf.predict([11.5,0.94]))


a.append(greycoprops(glcm3, 'dissimilarity')[0][0])
b.append(greycoprops(glcm3, 'correlation')[0][0])

p=a[0]
q=b[0]


print(clf.predict([p,q]))
'''
