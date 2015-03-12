import os
import subprocess as sp
import numpy as np

ANTSPATH = '/usr/local/antsbin/bin/'
BRATSPATH = '/home/brats/BRATS/Normalization/bin/'
input_path = '/mnt/windows/training/'
output_path = '/mnt/windows/MS_NormData/'
label_path = '/mnt/windows/MS_atropos/'
CSF = '/home/brats/BRATS/CSFMean/bin/CSFMean'
MeanValues = []

#Read patient list 
input_path, patients, files = os.walk(input_path).next()

#Make Masks
for p in patients:

    #Read all sequences
    timepoints = os.listdir(input_path+'/'+p+'/preprocessed')
    #timepoints = os.listdir(input_path+'/'+p)

    for t in timepoints:
        
        #imgs = os.listdir(input_path+'/'+p+'/preprocessed/'+t)
        imgs = os.listdir(input_path+'/'+p+'/preprocessed/'+t)

        #Make Output Dir
        if not os.path.exists(output_path+'/'+p+'/'+t):
            os.makedirs(output_path+'/'+p+'/'+t)

        #Prepare Mask
        #for i in imgs:
            
        #     input_file = input_path+'/'+p+'/preprocessed/'+t+'/'+i
        #     mask_file = output_path+'/'+p+'/'+t+'/mask.nii.gz'
            
        #     if ('t2' in i):               
        #         os.system(ANTSPATH+'ImageMath 3 '+mask_file+' Normalize '+input_file)
        #         os.system(ANTSPATH+'ThresholdImage 3 '+mask_file+' '+mask_file+' 0.02 1')
        #         os.system(ANTSPATH+'ImageMath 3 '+mask_file+' MD '+mask_file+' 1')
        #         os.system(ANTSPATH+'ImageMath 3 '+mask_file+' ME '+mask_file+' 1')
        #         os.system(ANTSPATH+'CopyImageHeaderInformation '+input_file+' '+mask_file+' '+mask_file+' 1 1 1')
    
        #Histogram Matching
        # for i in imgs:
        #     #input_file = input_path+'/'+p+'/preprocessed/'+t+'/'+i
        #     input_file = input_path+'/'+p+'/preprocessed/'+t+'/'+i
        #     output_file = output_path+'/'+p+'/'+t+'/N_'+i
            
        #     if ('mprage' in i):
        #         ref_image = '/mnt/windows/training/training01/preprocessed/01/training01_01_mprage_pp.nii'
        #     if ('t2' in i):
        #         ref_image = '/mnt/windows/training/training01/preprocessed/01/training01_01_t2_pp.nii'
        #     if ('pd' in i):
        #         ref_image = '/mnt/windows/training/training01/preprocessed/01/training01_01_pd_pp.nii'
        #     if ('flair' in i):
        #         ref_image = '/mnt/windows/training/training01/preprocessed/01/training01_01_flair_pp.nii'

        #     os.system(BRATSPATH+'Normalization '+output_file+' '+input_file+' '+ref_image)        


out1 = []
out2 = []
out3 = []
out4 = []

for p in patients:

    #Read all sequences
    timepoints = os.listdir(output_path+'/'+p)
    #timepoints = os.listdir(input_path+'/'+p)

    for t in timepoints:
        
        #imgs = os.listdir(input_path+'/'+p+'/preprocessed/'+t)
        imgs = os.listdir(output_path+'/'+p+'/'+t)

        #Make Output Dir

        #CSF Mean Normalisation
        for i in imgs:
            input_file = output_path+'/'+p+'/'+t+'/'+i
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

for p in patients:

    #Read all sequences
    timepoints = os.listdir(output_path+'/'+p)
    #timepoints = os.listdir(input_path+'/'+p)

    for t in timepoints:
        
        #imgs = os.listdir(input_path+'/'+p+'/preprocessed/'+t)
        imgs = os.listdir(output_path+'/'+p+'/'+t)

        for i in imgs:
            input_file = output_path+'/'+p+'/'+t+'/'+i
            output_file = output_path+'/'+p+'/'+t+'/'+i

            if ('mprage' in i):
                os.system(ANTSPATH+'ImageMath 3 '+output_file+' / '+input_file+' '+str(np.mean(out1)))
            if ('t2' in i):
                os.system(ANTSPATH+'ImageMath 3 '+output_file+' / '+input_file+' '+str(np.mean(out2)))
            if ('pd' in i):
                os.system(ANTSPATH+'ImageMath 3 '+output_file+' / '+input_file+' '+str(np.mean(out3)))
            if ('flair' in i):
                os.system(ANTSPATH+'ImageMath 3 '+output_file+' / '+input_file+' '+str(np.mean(out4)))

            mask_file = '/mnt/windows/MS_norm/'+p+'/'+t+'/mask.nii.gz'
            os.system(ANTSPATH+'ImageMath 3 '+output_file+' TruncateImageIntensity '+input_file+' 0.01 0.99 200')
            os.system(ANTSPATH+'ImageMath 3 '+output_file+' m '+mask_file+' '+output_file)