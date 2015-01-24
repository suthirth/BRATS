import itk
import os

image_type = itk.Image[itk.F,3]
reader = itk.ImageFileReader[image_type].New()

BRATSLIB_PATH = '/home/brats/BRATS/'

#Read files 
trpath = '/Dataset/'
trpath, patients, files = os.walk(trpath).next()

for p in patients:
    if not os.path.exists(trpath+p+'/Features'):
        os.makedirs(trpath+p+'/Features')
    imgs = os.listdir(trpath+'/'+p)
    for i in imgs:
        os.system(BRATSLIB_PATH+'ExtractFeatures/bin/ExtractFeatures '+trpath+p+'/ '+i+' '+trpath+p+'/Features/')

