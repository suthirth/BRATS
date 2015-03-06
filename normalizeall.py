import os

ANTSPATH = '/usr/local/antsbin/bin/'
input_path = '/mnt/windows/training/'
output_path = '/mnt/windows/MS_norm/'

#Read patient list 
input_path, patients, files = os.walk(input_path).next()

#Make Masks
for p in patients:

    #Read all sequences
    timepoints = os.listdir(input_path+'/'+p+'/preprocessed')

    for t in timepoints:
        
        imgs = os.listdir(input_path+'/'+p+'/preprocessed/'+t)

        #Make Output Dir
        if not os.path.exists(output_path+'/'+p+'/'+t):
            os.makedirs(output_path+'/'+p+'/'+t)

        #Prepare Mask
        for i in imgs:
            
            input_file = input_path+'/'+p+'/preprocessed/'+t+'/'+i
            mask_file = output_path+'/'+p+'/'+t+'/mask.nii.gz'
            
            if ('t2' in i):               
                os.system(ANTSPATH+'ImageMath 3 '+mask_file+' Normalize '+input_file)
                os.system(ANTSPATH+'ThresholdImage 3 '+mask_file+' '+mask_file+' 0.02 1')
                os.system(ANTSPATH+'ImageMath 3 '+mask_file+' MD '+mask_file+' 1')
                os.system(ANTSPATH+'ImageMath 3 '+mask_file+' ME '+mask_file+' 1')
                os.system(ANTSPATH+'CopyImageHeaderInformation '+input_file+' '+mask_file+' '+mask_file+' 1 1 1')
    
        #Intensity normalisation
        for i in imgs:
            input_file = input_path+'/'+p+'/preprocessed/'+t+'/'+i
            mask_file = output_path+'/'+p+'/'+t+'/mask.nii.gz'
            output_file = output_path+'/'+p+'/'+t+'/N_'+i
            os.system(ANTSPATH+'ImageMath 3 '+output_file+' TruncateImageIntensity '+input_file+' 0.01 0.99 200')
            os.system(ANTSPATH+'ImageMath 3 '+output_file+' m '+mask_file+' '+output_file)
            os.system(ANTSPATH+'ImageMath 3 '+output_file+' RescaleImage '+output_file+' 0 1')        
