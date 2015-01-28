import os

BRATSLIB_PATH = '/home/brats/BRATS/'

#Read files 
trpath = '/Test/'
trpath, patients, files = os.walk(trpath).next()

for p in patients:
    imgs = os.listdir(trpath+'/'+p)
    if not os.path.exists(trpath+p+'/Features'):
        os.makedirs(trpath+p+'/Features')
    for i in imgs:
        os.system(BRATSLIB_PATH+'ExtractFeatures/bin/ExtractFeatures '+trpath+p+'/ '+i+' '+trpath+p+'/Features/')

