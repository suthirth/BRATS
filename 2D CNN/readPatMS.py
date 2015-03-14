import os
import numpy
import nibabel as nib

IMAGES_PATH = '/mnt/windows/MS_NormData'
TRUTH_PATH = '/mnt/windows/MS_atropos/'

###############################################

class new():
	def __init__(self, pat_idx= None, time_idx = None):
		pat_str = str(pat_idx).zfill(2)
		time_str = str(time_idx).zfill(2)
		flair = nib.load(IMAGES_PATH + '/training' + pat_str + '/' + time_str + '/' + 'N_training' + pat_str + '_' + time_str + '_flair_pp.nii').get_data()
		t1 = nib.load(IMAGES_PATH + '/training' + pat_str + '/' + time_str + '/' + 'N_training' + pat_str + '_' + time_str + '_mprage_pp.nii').get_data()
		pd = nib.load(IMAGES_PATH + '/training' + pat_str + '/' + time_str + '/' + 'N_training' + pat_str + '_' + time_str + '_pd_pp.nii').get_data()
		t2 = nib.load(IMAGES_PATH + '/training' + pat_str + '/' + time_str + '/' + 'N_training' + pat_str + '_' + time_str + '_t2_pp.nii').get_data()
		self.truth = nib.load(TRUTH_PATH + '/training' + pat_str + '/' + time_str + '/' + 'truth7mask1.nii').get_data()
		self.data = numpy.array([flair,t1,pd,t2])