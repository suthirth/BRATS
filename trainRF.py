import itk
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

image_type = itk.Image[itk.US,3]
reader = itk.ImageFileReader[image_type].New()
itk_py_converter = itk.PyBuffer[image_type]

path = '/Dataset/'
truthpath = '/Truth'
path, patients, files = os.walk(path).next()

#Create output array
print 'Creating truth array...'
truth = np.zeros(155*240*240*len(patients))
for p in range(1,len(patients)):
	truthfile = os.listdir(truthpath+'/'+patients[p])
	reader.SetFileName(truthpath+'/'+patients[p]+'/'+truthfile[0])
	reader.Update()
	arr = np.array(itk_py_converter.GetArrayFromImage(reader.GetOutput()))
	arr = np.reshape(arr,155*240*240)
	truth[155*240*240*(p-1):155*240*240*p] = arr
print 'Truth array - done!'

seqindex = dict()
featureindex = dict()

seqindex = {'T1.':0,'T1C':1,'T2':2,'Flair':3}
featureindex = {'gauss_3':1,'gauss_7':2,'kurt_1':3,'kurt_3':4,'max_1':5,'max_3':6,'mean_1':7,'mean_3':8,'min_1':9,'min_3':10,'skw_1':11,'skw_3':12,'std_1':13,'std_3':14} 
#t1index = {'gauss_3':4,'gauss_7':5,'kurt_1':6,'kurt_3':7,'max_1':8,'max_3':9,'mean_1':10,'mean_3':11,'min_1':12,'min_3':13,'skw_1':14,'skw_3':15,'std_1':16,'std_3':17}
#t1cindex = {'gauss_3':18,'gauss_7':19,'kurt_1':20,'kurt_3':21,'max_1':22,'max_3':23,'mean_1':24,'mean_3':25,'min_1':26,'min_3':27,'skw_1':28,'skw_3':29,'std_1':30,'std_3':31}
#t2index = {'gauss_3':32,'gauss_7':33,'kurt_1':34,'kurt_3':35,'max_1':36,'max_3':37,'mean_1':38,'mean_3':39,'min_1':40,'min_3':41,'skw_1':42,'skw_3':43,'std_1':44,'std_3':45}
#flairindex = {'gauss_3':46,'gauss_7':47,'kurt_1':48,'kurt_3':49,'max_1':50,'max_3':51,'mean_1':52,'mean_3':53,'min_1':54,'min_3':55,'skw_1':56,'skw_3':57,'std_1':58,'std_3':59}

#Create feature array
print 'Creating feature array...'
features = np.zeros((155*240*240*len(patients),60))
for p in range(1,len(patients)):
	print 'Adding features of:', patients[p], '(',p,'/',len(patients),')' 
	seqfiles = os.listdir(path+'/'+patients[p])
	for s in seqfiles:
		if s!='Features':
			print path+'/'+patients[p]+'/'+s
			reader.SetFileName(path+'/'+patients[p]+'/'+s)
			reader.Update()
			arr = np.array(itk_py_converter.GetArrayFromImage(reader.GetOutput()))
			arr = np.reshape(arr,155*240*240)
			for si in seqindex:
				if si in s:
					features[155*240*240*(p-1):155*240*240*p,seqindex[si]] = arr

	featurefiles = os.listdir(path+'/'+patients[p]+'/Features')
	for f in featurefiles:
		print path+'/'+patients[p]+'/Features/'+f
		reader.SetFileName(path+'/'+patients[p]+'/Features/'+f)
		reader.Update()
		arr = np.array(itk_py_converter.GetArrayFromImage(reader.GetOutput()))
		arr = np.reshape(arr,155*240*240)
		for si in seqindex:
			if si in f:
				for fi in featureindex:
					if fi in f:
						features[155*240*240*(p-1):155*240*240*p,3+14*seqindex[si]+featureindex[fi]] = arr
	
print 'Feature array - done!'

print 'Creating Random Forest model...'
rf = RandomForestClassifier(n_estimators = 10);
rf.fit(features,truth)

joblib.dump(rf, 'randomforest.pkl') 
print 'RF done! Saved.'
#clf = joblib.load('randomforest.pkl') 