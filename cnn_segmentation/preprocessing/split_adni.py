import os
import glob
import numpy as np
from shutil import copy2


out_file_tr = './adni/training_mri.txt'
out_file_va = './adni/validation_mri.txt'
out_file_te = './adni/test_mri.txt'

out_dir_tr = './adni/train_mri/'
out_dir_va = './adni/vali_mri/'
out_dir_te = './adni/test_mri/'

out_dir_tr_leftH = './adni/train_leftH/'
out_dir_va_leftH = './adni/vali_leftH/'
out_dir_te_leftH = './adni/test_leftH/'

out_dir_tr_rightH = './adni/train_rightH/'
out_dir_va_rightH = './adni/vali_rightH/'
out_dir_te_rightH = './adni/test_rightH/'

# out_file_tr = '/old-home/rzn941/adni/training_mri.txt'
# out_file_va = '/old-home/rzn941/adni/validation_mri.txt'
# out_file_te = '/old-home/rzn941/adni/test_mri.txt'

# out_dir_tr = '/old-home/rzn941/adni/train_mri/'
# out_dir_va = '/old-home/rzn941/adni/vali_mri/'
# out_dir_te = '/old-home/rzn941/adni/test_mri/'

# out_dir_tr_leftH = '/old-home/rzn941/adni/train_leftH/'
# out_dir_va_leftH = '/old-home/rzn941/adni/vali_leftH/'
# out_dir_te_leftH = '/old-home/rzn941/adni/test_leftH/'

# out_dir_tr_rightH = '/old-home/rzn941/adni/train_rightH/'
# out_dir_va_rightH = '/old-home/rzn941/adni/vali_rightH/'
# out_dir_te_rightH = '/old-home/rzn941/adni/test_rightH/'

for out_dir in [out_dir_tr, out_dir_va, out_dir_te,
                out_dir_tr_leftH,out_dir_va_leftH, out_dir_te_leftH,
                out_dir_tr_rightH, out_dir_va_rightH, out_dir_te_rightH]:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

#
dir_mri = '/old-home/rzn941/code/skullStripB_tissueNormalized/mri/'
dir_lab_left = '/old-home/rzn941/code/skullStripB_tissueNormalized/label_L/'
dir_lab_right = '/old-home/rzn941/code/skullStripB_tissueNormalized/label_R/'

mri_files = glob.glob(dir_mri+'*.nii')
left_lab_files = glob.glob(dir_lab_left+'*.nii')
right_lab_files = glob.glob(dir_lab_right+'*.nii')

n_imgs = len(mri_files)
n_train = 25
n_vali = 5

rp = np.random.permutation(n_imgs)
mri_files = [mri_files[p] for p in rp]
train_files = mri_files[:n_train]
vali_files = mri_files[n_train:n_train+n_vali]
test_files = mri_files[n_train+n_vali:]

# copy the file and write the file into a file
with open(out_file_tr, 'w') as fout:
    for train_file in train_files:
        print("Copying training file {}".format(train_file))
        image_name, ext = os.path.splitext(train_file)
        # remove the parent directory
        image_name = os.path.split(image_name)[-1]
        # mri
        out_file = os.path.join(out_dir_tr, image_name+ext)
        copy2(train_file, out_file)
        # leftH
        out_file = os.path.join(out_dir_tr_leftH, image_name+ext)
        leftH_file = os.path.join(dir_lab_left, train_file[-24:-9]+'_L.nii')
        copy2(leftH_file, out_file)

        # rightH
        out_file = os.path.join(out_dir_tr_rightH, image_name+ext)
        rightH_file = os.path.join(dir_lab_right, train_file[-24:-9]+'_R.nii')
        copy2(rightH_file, out_file)
        fout.write(train_file+'\n')

with open(out_file_va, 'w') as fout:
    for vali_file in vali_files:
        print("Copying validation file {}".format(vali_file))
        image_name, ext = os.path.splitext(vali_file)
        image_name = os.path.split(image_name)[-1]
        out_file = os.path.join(out_dir_va, image_name+ext)
        copy2(vali_file, out_file)
        # leftH
        out_file = os.path.join(out_dir_va_leftH, image_name+ext)
        leftH_file = os.path.join(dir_lab_left, vali_file[-24:-9]+'_L.nii')
        copy2(leftH_file, out_file)

        # rightH
        out_file = os.path.join(out_dir_va_rightH, image_name+ext)
        rightH_file = os.path.join(dir_lab_right, vali_file[-24:-9]+'_R.nii')
        copy2(rightH_file, out_file)
        fout.write(vali_file+'\n')

with open(out_file_te, 'w') as fout:
    for test_file in test_files:
        print("Copying test file {}".format(test_file))
        image_name, ext = os.path.splitext(test_file)
        image_name = os.path.split(image_name)[-1]
        out_file = os.path.join(out_dir_te, image_name+ext)
        copy2(test_file, out_file)
        # leftH
        out_file = os.path.join(out_dir_te_leftH, image_name+ext)
        leftH_file = os.path.join(dir_lab_left, test_file[-24:-9]+'_L.nii')
        copy2(leftH_file, out_file)

        # rightH
        out_file = os.path.join(out_dir_te_rightH, image_name+ext)
        rightH_file = os.path.join(dir_lab_right, test_file[-24:-9]+'_R.nii')
        copy2(rightH_file, out_file)
        fout.write(test_file+'\n')

