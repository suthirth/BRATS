import os
import time

ANTSPATH = '/usr/local/antsbin/bin/'
input_path = '/mnt/windows/training'
norm_path = '/mnt/windows/MS_norm/'
output_path = '/mnt/windows/MS_atropos'
priors_path = '/home/brats/Kirby/Priors'

start = time.time()

#Read patient list 
input_path, patients, files = os.walk(input_path).next()

#Make Masks
for p in patients:

    #Read all sequences
    timepoints = os.listdir(input_path+'/'+p+'/preprocessed')

    for t in timepoints:
        imgs = os.listdir(input_path+'/'+p+'/preprocessed/'+t)

        #Make Output Dir
        output_base = output_path+'/'+p+'/'+t+'/'
        if not os.path.exists(output_base):
            os.makedirs(output_base)

        #Main loop
        for i in imgs:
            
            input_file = input_path+'/'+p+'/preprocessed/'+t+'/'+i
            mask_file = norm_path+'/'+p+'/'+t+'/mask.nii.gz'
            reg_file = output_base+'Reg'
            atropos_file = output_base+'atropos'
            output_file_prefix = output_base+'truth7'

            priors = os.listdir(priors_path)

            #Register intensity image, priors and run Atropos
            if ('mprage' in i):               
                os.system('bash '+ANTSPATH+'antsRegistrationSyN.sh -d 3 -f '+input_file+' -m ~/Kirby/S_template3_BrainCerebellum.nii.gz -o '+reg_file+' -t b -n 12')
                for pr in priors:

                    pr_input = priors_path+'/'+pr
                    pr_output = output_path+'/'+p+'/'+t+'/'+pr

                    os.system(ANTSPATH+'antsApplyTransforms -d 3 -i '+pr_input+' -o '+pr_output+' -r '+input_file+' -t '+reg_file+'1Warp.nii.gz -t '+reg_file+'0GenericAffine.mat -n BSpline')

                os.system('bash '+ANTSPATH+'antsAtroposN4.sh -d 3 -a '+input_file+' -x '+mask_file+' -c 6 -p '+output_base+'priors%d.nii.gz -o '+atropos_file+' w 0.25')

                #Locate Truth File and merge
                norm_files = os.listdir(norm_path+'/'+p+'/'+t+'/')
                for n in norm_files:
                    truth_file = norm_path+'/'+p+'/'+t+'/'+n
                    if 'mask1' in n:
                        output_file = output_file_prefix+'mask1.nii'
                        os.system(ANTSPATH+'ImageMath 3 '+output_file+' + '+truth_file+' 6')
                        os.system(ANTSPATH+'ImageMath 3 '+output_file+' m '+truth_file+' '+output_file)
                        os.system(ANTSPATH+'ImageMath 3 '+output_file+' overadd '+atropos_file+'Segmentation.nii.gz '+output_file)
                    if 'mask2' in n: 
                        output_file = output_file_prefix+'mask2.nii'
                        os.system(ANTSPATH+'ImageMath 3 '+output_file+' + '+truth_file+' 6')
                        os.system(ANTSPATH+'ImageMath 3 '+output_file+' m '+truth_file+' '+output_file)
                        os.system(ANTSPATH+'ImageMath 3 '+output_file+' overadd '+atropos_file+'Segmentation.nii.gz '+output_file)

end = time.time()
tt = end - start

print 'Time Elapsed: {}hrs, {}mins, {}secs.'.format(int(tt/3600),int((tt%3600)/60),int(tt%60)) 