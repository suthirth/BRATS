import itk
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

image_type = itk.Image[itk.F,3]
reader = itk.ImageFileReader[image_type].New()
itk_py_converter = itk.PyBuffer[image_type]

path = '/Dataset/'
truthpath = '/Truth'
path, patients, files = os.walk(path).next()


for p in patients:
	truthfiles = os.listdir(truthpath+'/'+p)
	for t in truthfiles:
		reader.SetFileName(truthpath+'/'+p+'/'+t)
		reader.Update()
		arr = np.array(itk_py_converter.GetArrayFromImage(reader.GetOutput()))
		arr = np.reshape(arr,155*240*240)
		np.append(Y,arr)

	featurefiles = os.listdir(path+'/'+p+'/Features')
	for f in featurefiles:
		if 'T1C' in f:
			if 'mean' in f:





rf = RandomForestClassifier(n_estimators = 200);
rf.fit(X,Y)

joblib.dump(rf, 'randomforest.pkl') 
#clf = joblib.load('randomforest.pkl') 

