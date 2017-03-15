"""
This script aims to convert a single multi-class segmentation
into several binary segmentations.

The source segmentation files should be processed before
so that they contain the degraded (weak) class list.

It exports the results to another file
so that you can delete it later to save space
"""
import os
import glob
import numpy as np
from scipy.io import loadmat, savemat
from spynet.utils.multiprocess import parmap

def export_bin_segs(pred_file):
    image_name = os.path.splitext(os.path.split(pred_file)[-1])[0]
    true_file = os.path.join(true_dir, image_name)+'_glm.mat'
    out_file = os.path.join(out_dir, image_name+'_glm.mat')

    true_seg = loadmat(true_file)
    true_labels = true_seg['label']

    pred_seg = loadmat(pred_file)
    pred_labels = pred_seg['label']
    weak_classes = pred_seg['degraded_classes'][0]
    print 'weak classes of {} = {}'.format(pred_file, weak_classes)
    if redo_export:
        # for each class, convert to binar (1/0)
        segs_shape = [len(weak_classes)]
        segs_shape.extend(pred_labels.shape)
        pred_binary_segs = np.zeros(segs_shape, dtype=np.int8)
        binary_diffs = np.zeros(segs_shape, dtype=np.int8)
        for i, c in enumerate(weak_classes): # 1 to 134
            bin_seg = np.zeros(pred_labels.shape, dtype=np.int8)
            bin_seg[np.where(pred_labels==c)] = 1

            true_bin_seg = np.zeros(true_labels.shape, dtype=np.int8)
            true_bin_seg[np.where(true_labels==c)] = 1

            pred_binary_segs[i,:,:,:] = bin_seg
            binary_diffs[i,:,:,:] = abs(bin_seg-true_bin_seg)

            # update the result
            pred_seg['binary_segmentations'] = pred_binary_segs
            pred_seg['binary_diffs'] = binary_diffs
        savemat(out_file, pred_seg)

out_dir = './datasets/miccai/test/bin_labels/'
true_dir = './datasets/miccai/test/label_mat/'
pred_dir = './experiments/6patchesCent_whole_brain_all/'
pred_files = glob.glob(pred_dir+'*.mat')
redo_export = True

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

parmap(export_bin_segs, pred_files)
# # sequential version below
# for pred_file in pred_files:
    # print 'Processing {}'.format(pred_file)
    # image_name = os.path.splitext(os.path.split(pred_file)[-1])[0]
    # out_file = os.path.join(out_dir, image_name+'_glm.mat')

    # pred_seg = loadmat(pred_file)
    # pred_labels = pred_seg['label']
    # weak_classes = pred_seg['degraded_classes'][0]

    # print 'weak classes={}'.format(weak_classes)
    # # for each class, convert to binar (1/0)
    # segs_shape = [len(weak_classes)]
    # segs_shape.extend(pred_labels.shape)
    # pred_binary_segs = np.zeros(segs_shape, dtype=np.int8)
    # for i, c in enumerate(weak_classes): # 1 to 134
        # print 'Export class {}'.format(c)
        # bin_seg = np.zeros(pred_labels.shape, dtype=np.int8)
        # bin_seg[np.where(pred_labels==c)] = 1
        # pred_binary_segs[i,:,:,:] = bin_seg

        # # update the result
        # pred_seg['binary_segmentations'] = pred_binary_segs
        # savemat(out_file, pred_seg)
