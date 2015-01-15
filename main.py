from medpy.io import load
import numpy as np
import itk
import scikit.learn 

#initialize ITK
image_type = itk.Image[itk.US,3]
reader = itk.ImageFileReader[image_type].New()
writer = itk.ImageFileWriter[image_type].New()

#Read files with MedPy/ ITK
trpath = 'Dataset/train'
trfiles = os.listdir(rootpath)
trdata = []
for f in trfiles:
#    data, header = load(trpath+'/'+f)
#    trdata.append(data)
    reader.SetFileName(trpath+'/'+f)
    reader.Update()
    trdata.append(reader.GetOutput())
    
print 'files:'+ len(trfiles) + 'on dir' + trpath
print 'data:' + np.shape(trdata[0])

#build mask and apply bias correction for all files
otsufilter = itk.OtsuThresholdImageFilter[image_type, image_type].New()
otsufilter.SetOutsideValue(255)
otsufilter.SetInsideValue(0)

n4filter = itk.N4BiasFieldCorrectionImageFilter[image_type, image_type, image_type].New()

maskeddata = []
for data in trdata:
	otsufilter.SetInput(data)
	otsufilter.Update()
	print 'Threshold value' + otsufilter.GetThreshold()
	maskeddata.append(otsufilter.GetOutput())


#mask 
	
#build neighborhood features

itk_py_converter = itk.PyBuffer[image_type]
arr = itk_py_converter.GetArrayFromImage(image)

#



