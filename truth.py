import os

## Variable settings here ### 
ANTSPATH = '/usr/local/antsbin/bin'
INPUT_DIR = '/Truth'
TEMPLATE_DIR = 
OUTPUT_DIR = '/mnt/windows/Atropos_Truth/'

## Processing here
input_path, patients, files = os.walk(input_path).next()

for p in patients:
    
    #Read all sequences
    imgs = os.listdir(input_path+'/'+p)

    #Make Output Dir
    if not os.path.exists(output_path+'/'+p):
        os.makedirs(output_path+'/'+p)

    #Prepare Mask
    mask_file = output_path+'/'+p+'/MASK.mha'
    for i in imgs:
        if ('T2' in i):
            print i
            os.system(ANTSPATH+'ImageMath 3 '+mask_file+' Normalize '+input_path+'/'+p+'/'+i)
            os.system(ANTSPATH+'ThresholdImage 3 '+mask_file+' '+mask_file+' 0.1 1')