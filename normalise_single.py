import os

ANTSPATH = '/usr/local/antsbin/bin/'
input_file = '/home/brats/VSD.Brain.XX.O.MR_T1.35536.mha'
mask_file = '/home/brats/VSD.Brain.Mask.mha'
output_file = '/home/brats/VSD.MR_T1.35535_Normalized.mha'
prepare_mask = 1 #1 for yes, 0 for no

#Prepare Mask
if (prepare_mask==1):
    os.system(ANTSPATH+'ImageMath 3 '+mask_file+' Normalize '+input_file)
    os.system(ANTSPATH+'ThresholdImage 3 '+mask_file+' '+mask_file+' 0.1 1')
    os.system(ANTSPATH+'ImageMath 3 '+mask_file+' MD '+mask_file+' 3')
    os.system(ANTSPATH+'ImageMath 3 '+mask_file+' ME '+mask_file+' 3')

#Intensity normalisation and/or N4 Bias
os.system(ANTSPATH+'ImageMath 3 '+output_file+' TruncateImageIntensity '+input_file+' 0.01 0.99 200')
#os.system(ANTSPATH+'N4BiasFieldCorrection -d 3 -c[20x20x20x10,0] -x '+mask_file+' -b [200] -s 2 -i '+output_file+' -o '+output_file)
os.system(ANTSPATH+'ImageMath 3 '+output_file+' m '+mask_file+' '+output_file)
os.system(ANTSPATH+'ImageMath 3 '+output_file+' RescaleImage '+output_file+' 0 1')        