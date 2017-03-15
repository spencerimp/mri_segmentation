"""
A pipleline that runs three modules
1. training
2. prediction
3. performance assessment.
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
from train_triplanar_cent import (
    train_triplanar_cent,
)
from segmentation_triplanar_cent import segment_triplanar_cent
from assess_segmentation import compute_performance


# pipeline configs
nib.nifti1.Nifti1Header.quaternion_threshold = -1e-6
out_process_time = './experiments/keras/triplanar_cent/process_time.txt'
is_reset_exp = True

# training configs
train_img_dir = './datasets/miccai/train/mri/'
train_lab_dir = './datasets/miccai/train/label/'
model_path = './experiments/keras/triplanar_cent/cnn_triplanar_cent.h5'
stats_path = './experiments/keras/triplanar_cent/cnn_triplanar_cent_stat.h5'
logger_path = './experiments/keras/triplanar_cent/train_triplanar_cent_log.csv'
n_classes = 135
patch_size = 29
scales = [1, 3]
n_voxels_tr = 60000
n_voxels_va = 10000
train_batch_size = 200
max_epoch = 300
init = 'he_uniform'
extract_parallel = True
optimizer = SGD(lr=0.05, momentum=0.5, nesterov=False)
# optimizer = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
# optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
# optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                 # epsilon=1e-08, decay=0.0)
# prediction configs
test_img_dir = './datasets/miccai/test/mri/'
# mask_dir = './datasets/miccai/test/label_mat/' #  cheating centroids
mask_dir = './experiments/keras/triplanar/'  # from patch-only prediction
prediction_worker = './prediction_worker_triplanar_cent.py'
generator_batch_size = 200000
gpu_batch_size = 50000
out_segmentation_dir = './experiments/keras/triplanar_cent/'
gpus = ['gpu0', 'gpu1', 'gpu2']

# assessment configs
performance_file = './experiments/keras/triplanar_cent/performance_triplanar_cent.csv'
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

train_triplanar_cent(train_img_pathes, train_lab_pathes,
                     model_path, stats_path, logger_path,
                     n_classes, patch_size, scales,
                     n_voxels_tr, n_voxels_va,
                     train_batch_size, max_epoch,
                     optimizer, init, extract_parallel, is_reset_exp)

train_time = time.time() - start_time
print("Training done in {} seconds".format(train_time))

# parse and run segmentation
start_time = time.time()
test_img_pathes = []
test_mask_pathes = []
for img_path in glob.glob(test_img_dir+'*.nii'):
    out_path = change_parent_dir(out_segmentation_dir, img_path, '_glm.mat')
    # only process those images without results
    if not os.path.exists(out_path):
        mask_path = change_parent_dir(mask_dir, img_path, '_glm.mat')
        test_img_pathes.append(img_path)
        test_mask_pathes.append(mask_path)

segment_triplanar_cent(test_img_pathes, test_mask_pathes,
                       prediction_worker,
                       model_path, stats_path,
                       n_classes, patch_size, scales,
                       generator_batch_size, gpus, gpu_batch_size,
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
