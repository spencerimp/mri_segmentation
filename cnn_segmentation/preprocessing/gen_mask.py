"""
This script generates a binary mask for each test images.
This script calls Brain Extraction Tool (BET)
to perform skull stripping, and convert the brain scan
into binary matrix.

I the BET cannot be performed, you should check
    ../tools/README.md
to install the packages and interfaces.
"""
import os
import glob
import numpy as np
import nibabel as nib
import nipype.interfaces.fsl as fsl
import scipy
from scipy.io import savemat
from utils.utils import (
    change_parent_dir,
    load_mri,
)


nib.nifti1.Nifti1Header.quaternion_threshold = -1e-6
test_img_dir = './datasets/miccai/test/mri/'
test_mask_dir = './datasets/miccai/test/mask/'
dim = 3

if not os.path.exists(test_mask_dir):
    os.makedirs(test_mask_dir)

test_imgs = glob.glob(test_img_dir+'*.nii')
for test_path in test_imgs:
    crop_path = change_parent_dir(test_mask_dir, test_path, '.nii')
    mybet = fsl.BET(in_file=test_path, out_file=crop_path)
    mybet.run()

    mask_path = change_parent_dir(test_mask_dir, crop_path, '_glm.mat')
    img = load_mri(test_path)
    # convert the intensity to binary mask
    mask = scipy.sign(img)

    data = {}
    data['label'] = mask
    savemat(mask_path, data)
    print("Save mask to {}".format(mask_path))
    os.remove(crop_path)
