import nrrd # pip install pynrrd
import nibabel as nib # pip install nibabel
from pathlib import Path
import numpy as np
import os
for filename in Path('LIDC_conversion5').glob('**/*.nrrd'):
	filename_nii = str(filename) + '.nii.gz'
	if not os.path.isfile(filename_nii):
		print(filename)
		_nrrd = nrrd.read(filename)
		data = _nrrd[0]
		img = nib.Nifti1Image(data, np.eye(4))
		nib.save(img, filename_nii)
