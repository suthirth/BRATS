import os
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import itk

path = '/Dataset/brats_tcia_pat105_1'

seqindex = {'T1.':0,'T1C':1,'T2':2,'Flair':3}
featureindex = {'gauss_3':1,'gauss_7':2,'kurt_1':3,'kurt_3':4,'max_1':5,'max_3':6,'mean_1':7,'mean_3':8,'min_1':9,'min_3':10,'skw_1':11,'skw_3':12,'std_1':13,'std_3':14} 

features = np.zeros((155*240*240,60))
print 'Building feature array' 
seqfiles = os.listdir(path)
for s in seqfiles:
		if s!='Features':
			reader.SetFileName(path+'/'+patients[p]+'/'+s)
			reader.Update()
			arr = np.array(itk_py_converter.GetArrayFromImage(reader.GetOutput()))
			arr = np.reshape(arr,155*240*240)
			for si in seqindex:
				if si in s:
					features[0:155*240*240,seqindex[si]] = arr

featurefiles = os.listdir(path+'/Features')
for f in featurefiles:
	print path+'/Features/'+f
	reader.SetFileName(path+'/Features/'+f)
	reader.Update()
	arr = np.array(itk_py_converter.GetArrayFromImage(reader.GetOutput()))
	arr = np.reshape(arr,155*240*240)
	for si in seqindex:
		if si in f:
			for fi in featureindex:
				if fi in f:
					features[0:155*240*240,3+14*seqindex[si]+featureindex[fi]] = arr

rf = joblib.load('/RF/randomforest.pkl')
pred = rf.predict(features)

x = pred.reshape(155,240,240)
