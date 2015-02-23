import os

ANTSPATH = '/usr/local/antsbin/bin/'
input_path = '/home/brats/dataset'
output_path = '/home/brats/norm_dataset'

#Read patient list 
input_path, patients, files = os.walk(input_path).next()

for p in patients:
    
    #Read all sequences
    imgs = os.listdir(input_path+'/'+p)

    #Make Output Dir
    if not os.path.exists(output_path+'/'+p):
        os.makedirs(output_path+'/'+p)

    #Prepare Mask
    mask_file = output_path+'/'+p+'/MASK.mha'
    os.system(ANTSPATH+'ImageMath 3 '+mask_file+' Normalize '+input_path+'/'+p+'/'+imgs[0])
    os.system(ANTSPATH+'ThresholdImage 3 '+mask_file+' '+mask_file+' 0.1 1')

    #Intensity normalisation and/or N4 Bias
    for i in imgs:
    	input_file = input_path+'/'+p+'/'+i
    	output_file = output_path+'/'+p+'/NORMALIZED_'+i
        os.system(ANTSPATH+'ImageMath 3 '+output_file+' TruncateImageIntensity '+input_file+' 0.01 0.99 200')
    #    os.system(ANTSPATH+'N4BiasFieldCorrection -d 3 -c[20x20x20x10,0] -x '+mask_file+' -b [200] -s 2 -i '+ouput_file+' -o '+output_file)
		os.system(ANTSPATH+'ImageMath 3 '+output_file+' m '+mask_file+' '+output_file)
		os.system(ANTSPATH+'ImageMath 3 '+output_file+' RescaleImage '+output_file+' 0 1')        