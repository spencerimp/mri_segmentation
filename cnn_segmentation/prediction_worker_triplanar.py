"""
This worker script segment a list of unseen images.
It can be used as the worker scripts so that
the manager script can segment multiple images concurrently.

This only works for CnnTriplanarMultiset network
"""
import os
import sys
import nibabel as nib
from scipy.io import savemat
from network import CnnTriplanarMultiset
from utils.voxel_feature import PickTriplanarMultiPatch
from cnn_utils import segment_by_generator


def param_wrapper(param):
    """A wrapper that adds single quotes to the elements of parameters.

    >>> param_wrapper('[file1, file2]')
    # ['file1', 'file2']
    """
    file_list = []
    elem = ''
    for c in param:
        if c not in ['[', ']', ',', ' ']:
            elem += c
        else:
            file_list.append(elem)
            elem = ''

    return file_list


def segment_triplanar(img_pathes, mask_pathes,
                      model_path, stats_path,
                      n_classes, patch_size, scales,
                      batch_size, gpu_batch_size,
                      out_dir):
    n_channels = 3 * len(scales)
    net = CnnTriplanarMultiset(patch_size=patch_size,
                               n_channels=n_channels,
                               out_size=n_classes)
    net.load_model(model_path)
    net.load_stats(stats_path)
    patch_picker_class = PickTriplanarMultiPatch
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    for img_path, mask_path in zip(img_pathes, mask_pathes):
        img_name = os.path.split(os.path.splitext(mask_path)[0])[-1]

        # store the raw output of non-background voxels
        prob_path = os.path.join(out_dir, 'prob', img_name) + '.mat'
        prob_dir = os.path.dirname(prob_path)
        os.makedirs(prob_dir, exist_ok=True)

        # set final output path
        out_path = os.path.join(out_dir, img_name) + '.mat'
        pred_lab = segment_by_generator(net, img_path, mask_path,
                                        patch_size, scales,
                                        patch_picker_class,
                                        batch_size, gpu_batch_size,
                                        prob_path=prob_path)

        data = {}
        data['label'] = pred_lab
        print("Save segmentatation to {}".format(out_path))
        savemat(out_path, data)


if __name__ == '__main__':
    nib.nifti1.Nifti1Header.quaternion_threshold = -1e-6
    img_pathes = param_wrapper(sys.argv[1])[1:]
    mask_pathes = param_wrapper(sys.argv[2])
    model_path = param_wrapper(sys.argv[3])[-1]
    stats_path = param_wrapper(sys.argv[4])[-1]
    n_classes = int(param_wrapper(sys.argv[5])[-1])
    patch_size = int(param_wrapper(sys.argv[6])[-1])
    scales = list(map(int, param_wrapper(sys.argv[7])))
    batch_size = int(param_wrapper(sys.argv[8])[-1])
    gpu_batch_size = int(param_wrapper(sys.argv[9])[-1])
    out_dir = param_wrapper(sys.argv[10])[-1]

    # only operate if the list has something
    segment_triplanar(img_pathes, mask_pathes,
                      model_path, stats_path,
                      n_classes, patch_size, scales,
                      batch_size, gpu_batch_size,
                      out_dir)
