import numpy as np
import math,os
from sklearn import mixture
import matplotlib.pyplot as plt
from scipy.io import loadmat 
import Image

rootpath = 'BRATSMAT'
image_data = loadmat(rootpath+'/1t.mat')
#print image_data

classes = [0,1,2,3,4]
#0 - everything else
#1 - necrosis
#2 - edema
#3 - non enhancing
#4 - enhancing

classdata = [[] for i in range(5)]
for i in range(1,4):
    image_data = loadmat(rootpath+'/'+str(i)+'.mat')
    truth_data = loadmat(rootpath+'/'+str(i)+'t.mat')         
    imagedata = np.reshape(image_data['intensityValues'],8928000,1)
    truthdata = np.reshape(truth_data['truthValues'],8928000,1)    
    print np.shape(imagedata)    
    for j in range(8928000):
        #print truthdata[j]
        classdata[truthdata[j]].append(imagedata[j])
    print np.shape(classdata[0]), np.shape(classdata[1]), np.shape(classdata[2]), np.shape(classdata[3]), np.shape(classdata[4])               
print np.shape(classdata[0]), np.shape(classdata[1]), np.shape(classdata[2])

#Train the GMMs for each class
GMMs = []
n_clusters = [4,2,2,2,2]
for i in classes:
    print "Training GMM for", i
    GMMs.append(mixture.GMM(n_clusters[i]))
    #print np.shape(classdata[i][:1000])
    GMMs[i].fit(classdata[i][:1000000])
    #print GMMs[i].get_params()
    #GMMs[i].saveloglikelihood('likelihood'+str(i))
           
#Use GMMs for testing            
test_data = loadmat(rootpath+'/1.mat')
testdata = np.reshape(test_data['intensityValues'][96,:,:],37200,1)

posterior = []
for j in range(5):
    p = GMMs[j].predict_proba(testdata)
    posterior.append(np.sum(p,axis=1))
#print 'posterior', posterior[0]

prediction = []
for j in range(37200):
    post = [posterior[i][j] for i in range(5)]
    print np.shape(posterior[i][j])    
    prediction.append(post.index(np.max(post)))

print np.shape(prediction)
predict_data = np.reshape(np.array(prediction),(155,240))
print np.shape(predict_data), predict_data
imgdata = 63.0 * predict_data.astype(np.uint8)
print imgdata
img = Image.fromarray(imgdata)
img.show()    
#plt.imshow(predict_data) 
#plt.savefig('output.png')


