import os
import numpy as np
import glob
from utils.utils import (
    load_miccai_labels,
    change_parent_dir,
)
from scipy.io import savemat

lab_dir = './datasets/miccai/test/label_mat/'
mask_dir = './datasets/miccai/test/label_mat_hippo/'

if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

lab_pathes = glob.glob(lab_dir+'*.mat')
for lab_path in lab_pathes:
    out_path = change_parent_dir(mask_dir, lab_path, '.mat')
    lab = load_miccai_labels(lab_path)
    lab = lab.astype(int)

    # 17 (right hippo) -> 1
    # 18 (left hippo) -> 2

    lab[lab == 17] = 1000
    lab[lab == 18] = 2000
    lab[np.where((lab != 1000) & (lab != 2000))] = 0
    lab[lab == 1000] = 1
    lab[lab == 2000] = 2

    data = {}
    data['label'] = lab
    savemat(out_path, data)
