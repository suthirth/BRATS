import itk
import os

#initialize ITK
image_type = itk.Image[itk.UC,3]
reader = itk.ImageFileReader[image_type].New()
writer = itk.ImageFileWriter[image_type].New()

#Read files 
trpath = '/home/brats/BRATS_data/Train'
trfiles = os.listdir(trpath)
trdata = []
for f in trfiles:
    reader.SetFileName(trpath+'/'+f)
    reader.Update()
    trdata.append(reader.GetOutput())
    
print 'files:'+ len(trfiles) + 'on dir' + trpath
print 'data:' + np.shape(trdata[0])

#build mask and apply bias correction for all files
otsufilter = itk.OtsuThresholdImageFilter[image_type, image_type].New()
otsufilter.SetOutsideValue(1)
otsufilter.SetInsideValue(0)

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



