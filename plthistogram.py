import nibabel as nib
import matplotlib.pyplot as plt 
import numpy as np
import sys

if len(sys.argv) <2:
	print 'Put file name bro.'

img = nib.load(sys.argv[1])
data = img.get_data()
size = data.shape[0]* data.shape[1]* data.shape[2]
data = np.reshape(data, size)
data = [d for d in data if d!=0]

plt.hist(data, 200)
plt.show()