cd $ANTSPATH
./ImageMath 3 ~/MS_1/label.nii + ~/MS_1/training01_01_mask1.nii 6
./ImageMath 3 ~/MS_1/label.nii m ~/MS_1/training01_01_mask1.nii ~/MS_1/label.nii 
./ImageMath 3 ~/MS_1/label1.nii overadd ~/MS_1/Atropos/6cw25Segmentation.nii.gz ~/MS_1/label.nii 

