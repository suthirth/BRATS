import subprocess as sp
import os
import numpy as np

CSF = '/home/brats/BRATS/CSFMean/bin/CSFMean'
input_path = '/mnt/windows/MS_norm2/'
label_path = '/mnt/windows/MS_atropos/'

#Read patient list 
input_path, patients, files = os.walk(input_path).next()

out1 = []
out2 = []
out3 = []
out4 = []

#Make Masks
for p in patients:

    #Read all sequences
    timepoints = os.listdir(input_path+'/'+p)

    for t in timepoints:
        
        imgs = os.listdir(input_path+'/'+p+'/'+t)

        #Intensity normalisation
        for i in imgs:
            input_file = input_path+'/'+p+'/'+t+'/'+i
            label_file = label_path+'/'+p+'/'+t+'/truth7mask1.nii'
            proc = sp.Popen([CSF+' '+input_file+' '+label_file+' 1'], stdout=sp.PIPE, shell=True)
            (out, err) = proc.communicate()

            if ('mprage' in i):
                out1.append(float(out))
            if ('t2' in i):
                out2.append(float(out))
            if ('pd' in i):
                out3.append(float(out))
            if ('flair' in i):
                out4.append(float(out))

print "Mean of T1, T2, PD, FLAIR:"
print np.mean(out1), np.mean(out2), np.mean(out3), np.mean(out4)
