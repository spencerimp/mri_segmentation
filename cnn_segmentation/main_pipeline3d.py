"""
A pipleline that runs three modules
1. training
2. prediction
3. performance assessment.

This script uses triplanar patches with multiple scales as input.
"""
import os
import sys
import time
import glob
import shutil
import nibabel as nib
from keras.optimizers import (
    SGD,
    Adam,
    Adagrad,
    Adadelta,
)
from utils.utils import change_parent_dir
from train3d import (
    train3DPatch,
)
from segmentation3d import segment_images3DPatch
from assess_segmentation import compute_performance


# pipeline configs
nib.nifti1.Nifti1Header.quaternion_threshold = -1e-6
out_process_time = './experiments/keras/3DPatch/process_time.txt'
is_reset_exp = True

# training configs
train_img_dir = './datasets/miccai/train/mri/'
train_lab_dir = './datasets/miccai/train/label/'
model_path = './experiments/keras/3DPatch/cnn_3DPatch.h5'
stats_path = './experiments/keras/3DPatch/cnn_3DPatch_stat.h5'
logger_path = './experiments/keras/3DPatch/train3DPatch_log.csv'
n_classes = 135
patch_size = 13
scales = [1, 3]
n_voxels_tr = 60000
n_voxels_va = 10000
train_batch_size = 200
max_epoch = 300
init = 'he_uniform'
extract_parallel = True
# extract_parallel = False
optimizer = SGD(lr=0.05, momentum=0.5, decay=1e-6, nesterov=False)
# optimizer = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
# optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
# optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                 # epsilon=1e-08, decay=0.0)
# prediction configs
test_img_dir = './datasets/miccai/test/mri/'
mask_dir = './datasets/miccai/test/label_mat/'
prediction_worker = './prediction_worker3d.py'
generator_batch_size = 80000
gpu_batch_size = 8000
out_segmentation_dir = './experiments/keras/3DPatch/'
gpus = ['cuda0', 'cuda1', 'cuda2', 'cuda3']

# assessment configs
performance_file = './experiments/keras/3DPatch/performance3DPatch.csv'
true_label_dir = './datasets/miccai/test/label_mat/'
verbose = False
label_list_file = './docs/MICCAI-Challenge-2012-Label-Information_v3.csv'

# parse and run training
start_time = time.time()
if is_reset_exp:
    shutil.rmtree(out_segmentation_dir, ignore_errors=True)

train_img_pathes = glob.glob(train_img_dir+'*.nii')
train_lab_pathes = [change_parent_dir(train_lab_dir, train_img_path, '_glm.nii')
                    for train_img_path in train_img_pathes]

train3DPatch(train_img_pathes, train_lab_pathes,
              model_path, stats_path, logger_path,
              n_classes, patch_size, scales,
              n_voxels_tr, n_voxels_va,
              train_batch_size, max_epoch,
              optimizer, init,
              extract_parallel, is_reset_exp)

train_time = time.time() - start_time
print("Training done in {} seconds".format(train_time))

# parse and run segmentation
start_time = time.time()
# only process those images without results
test_img_pathes = []
for img_path in glob.glob(test_img_dir+'*.nii'):
    out_path = change_parent_dir(out_segmentation_dir, img_path, '.mat')
    if not os.path.exists(out_path):
        test_img_pathes.append(img_path)

segment_images3DPatch(test_img_pathes, mask_dir,
                       prediction_worker,
                       model_path, stats_path,
                       n_classes, patch_size, scales,
                       generator_batch_size,
                       gpus, gpu_batch_size,
                       out_segmentation_dir)

segment_time = time.time() - start_time
print("Segmentation done in {} seconds".format(segment_time))

# assessment
pred_label_files = glob.glob(out_segmentation_dir+'*.mat')
true_label_files = [change_parent_dir(true_label_dir, f, '.mat')
                    for f in pred_label_files]
compute_performance(pred_label_files,
                    true_label_files,
                    n_classes,
                    performance_file,
                    verbose,
                    label_list_file)

with open(out_process_time, 'a') as fout:
    fout.write("Training time, {}\n".format(train_time))
    fout.write("Segmentation time, {}\n".format(segment_time))

# copy this script and network (source files)
shutil.copy2(sys.argv[0], out_segmentation_dir)
shutil.copy2('./network.py', out_segmentation_dir)
print("Experiment done!")
