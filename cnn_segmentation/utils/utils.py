import os
import csv
import glob
import h5py
import shutil
import random
import numpy as np
import nibabel as nib
import multiprocessing
from multiprocessing import Pool
from joblib import Parallel, delayed
from scipy.io import loadmat
from scipy.ndimage import label as ndlabel
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def label_filtering(lab, ignored_labels, true_labels):
    """Convert the labels and replace nan and inf with zeros.

    The filtered label starts from 1.
    """
    lab[np.isnan(lab)] = 0
    lab[np.isinf(lab)] = 0

    # skip if the labels are already correct
    # (a strong assumption that we always have the largest label)
    if np.max(lab.ravel()) <= len(true_labels):
        return lab

    for ignored_label in ignored_labels:
        lab[lab == ignored_label] = 0
    for idx, label in enumerate(true_labels):
        lab[lab == label] = idx + 1
    return lab


def get_nonzero_limit(img, axis, idx_range):
    """Get the index first hyperplane containing nonzeros.

    Input:
        img (np.ndarray): tensor, could be 2d or 3d
        axis (int): the axis to scan
        idx_range (list-like): ordered indice to search

    Output:
        the first index at which contains a nonzero hyperplane.
    """
    dim = len(img.shape)
    s = [slice(None)] * dim

    # scan every plane until a nonzero item is found
    for idx in idx_range:
        # the plane cuts through this point
        s[axis] = idx
        if img[s].any():
            return idx

    # otherwise, return the last index
    return idx_range[-1]


def get_largest_component(lab):
    """Get largest connected component.

    Given a multi-class labeling,
    leave the largest connected component
    for each class.
    """
    classes = np.unique(lab)
    classes = np.delete(classes, np.argwhere(classes == 0))
    pruned_lab = np.zeros(lab.shape, dtype=lab.dtype)
    for c in classes:
        print("Finding largest connected component in class {}".format(c))
        # make it black and white
        bw = np.zeros(lab.shape)
        bw[lab == c] = 1
        # 26 connectivity for 3D images
        conn = np.ones((3,3,3))
        # clustered_lab.shape = bw.shape
        clustered_lab, n_comps = ndlabel(bw, conn)
        # sort components by volume from smallest to largest (skip zero)
        comp_volumes = [np.sum(clustered_lab == i) for i in range(1, n_comps)]
        comp_labels = np.argsort(comp_volumes)
        # pick component with largest volume (not counting background)
        largest_comp_label = 1 + comp_labels[-1]

        # keep the component in the output
        pruned_lab[clustered_lab==largest_comp_label] = c
    return pruned_lab


def get_boundary(img):
    """Get the boundary of non-zero region.

    Input:
        img (np.ndarray): image, could be 2d or 3d

    Output:
        The limit tuples of each axis
         (i.e. min voxel and max voxel)
    """
    img_shape = img.shape
    dim = len(img_shape)

    # get boundary to each axis
    boundary = np.zeros((dim, 2), dtype=int)
    for ax in range(dim):
        # forward to get minimum
        ax_min = get_nonzero_limit(img, ax, range(img_shape[ax]))
        # backward to get maximum
        ax_max = get_nonzero_limit(img, ax, reversed(range(img_shape[ax])))
        boundary[ax] = [ax_min, ax_max]
    return boundary


def crop_images(img, mask):
    """Crop image and mask."""
    # crop
    dim = len(img.shape)
    nz_limit = get_boundary(mask)

    s = [None] * dim
    for axis in range(dim):
        idx_min = nz_limit[axis, 0]
        idx_max = nz_limit[axis, 1]
        s[axis] = slice(idx_min, idx_max + 1)  # idx_max should be included
    img = img[s]
    mask = mask[s]
    return img, mask, nz_limit


def pad_images(pad_width, *imgs):
    """Pad zeros to the boundaries."""
    dim = len(imgs[0].shape)
    # if scale, then share with each axis
    if type(pad_width) == int:
        pad_width = tuple([pad_width] * dim)
    # tuple (beginning_width, end_width) for each axis
    # same width for beginning and end
    pad_widthes = tuple(zip(pad_width, pad_width))
    padded_imgs = []
    for i, img in enumerate(imgs):
        padded_img = np.lib.pad(img, pad_widthes, 'constant', constant_values=0)
        padded_imgs.append(padded_img)
    return padded_imgs


def convert_voxels_padding(voxels, pad_width):
    """
    Convert the voxels gained before padding
    to the voxels after padding.
    (i.e. position at small image -> position at large image).
    """
    dim = voxels.shape[1]
    new_voxels = voxels.copy()
    if type(pad_width) == int:
        pad_width = tuple([pad_width] * dim)
    for axis in range(dim):
        new_voxels[:, axis] += pad_width[axis]
    return new_voxels


def convert_voxels_cropped(voxels, pad_width):
    """
    Convert the voxels gained after padding
    to voxels before padding.
    (i.e. position at large image -> position at small image).
    """
    dim = voxels.shape[1]
    new_voxels = voxels.copy()
    if type(pad_width) == int:
        pad_width = tuple([pad_width] * dim)
    for axis in range(dim):
        new_voxels[:, axis] -= pad_width[axis]
    return new_voxels


def convert_voxels_original(voxels, nz_limit):
    """
    Convert the voxels gained after cropped
    to voxels before crooped
    (i.e. position at cropped image -> position at original image).
    """
    dim = voxels.shape[1]
    new_voxels = voxels.copy()
    for axis in range(dim):
        idx_min = nz_limit[axis][0]
        new_voxels[:, axis] += idx_min
    return new_voxels


def recover_image(vxs, labels, orig_shape):
    """Recover the cropped image to origin size.

    Inputs:
        vxs: numpy.ndarray, shape = (n_voxels, 3)
            The voxels on original image
        labels: numpy.ndarray, shape = (n_voxels, )
            The corresponding label of the voxels
        orig_shape: the shape of original image
    """
    orig_labels = np.zeros(orig_shape, dtype=labels.dtype)
    orig_labels[vxs[:, 0], vxs[:, 1], vxs[:, 2]] = labels.ravel()
    return orig_labels


def get_coordinates(img_shape):
    """Get voxels (or pixels) given the image shape."""
    dim = len(img_shape)
    cords = None
    if dim == 2:  # 2D image
        cords = [(x, y) for x in range(img_shape[0])
                 for y in range(img_shape[1])]
    elif dim == 3:
        cords = [(x, y, z) for x in range(img_shape[0])
                 for y in range(img_shape[1])
                 for z in range(img_shape[2])]
    return np.asarray(cords)


def get_nonzero_voxels(labels):
    nz = np.where(labels != 0)
    return np.array(list(zip(nz[0], nz[1], nz[2])))


def zscore(X, **kwargs):
    """Apply zscore to matrix X.

    Given a matrix X (row major)
    normalize the features to make them have mean of zero and std of 1

    You can also assign the mean (mu) and std (sigma) by

    >>> zscore(X, mu=m, sigma=s)

    Then X will be normalzed using m and s instead of its own statistics.
    Returns:
            1. normalized X
            2. used mean when normalizing X
            3. used std when normalizing X
    """
    # fetch the mu and std if any
    meanX = np.mean(X, axis=0)
    stdX = np.std(X, axis=0)
    if 'mu' in kwargs:
        meanX = kwargs.get('mu')
    if 'sigma' in kwargs:
        stdX = kwargs.get('sigma')

    X -= meanX
    X /= stdX
    return X, meanX, stdX


def image_smooth(img, lab):
    """Smooth an image by the mean and std of the nonzero items.
    """
    nonzero = lab.nonzero()
    mu = np.mean(img[nonzero])
    sigma = np.std(img[nonzero])
    return zscore(img, mu=mu, sigma=sigma)[0]


def get_label_distribution(labels, regions=None):
    """Return the label distribution."""
    n_all_samples = len(labels)
    if not regions:
        regions = np.unique(labels)
    dist = np.zeros((len(regions),), np.float32)
    for i, region in enumerate(regions):
        n_samples = len(np.where(labels == region)[0])
        dist[i] = n_samples / float(n_all_samples)
    return dist


def shuffle_voxels(voxels):
    rp = np.random.permutation(voxels.shape[0])
    return voxels[rp]


def shuffle_all(*data_list):
    """Shuffle all the arrays together.

    Input:
        arbitrary number of data list
    Output:
        shuffled data_list
    Example:
        indices = [0,1,2,3]
        values = [100,101,102,103]
        indices, values = shuffle_all(indices, values)
        # indices = [2,1,0,3]
        # values = [102,101,100,103]
    """
    len_data = len(data_list[0])
    rp = np.random.permutation(len_data)
    data_out = []
    for i, data in enumerate(data_list):
        assert len(data) == len_data, "The length of each data should be equal"
        data_out.append(data[rp])

    if len(data_list) == 1:
        data_out = data_out[0]
    return data_out


def parfor(fun, args, n_jobs=multiprocessing.cpu_count()):
    """Run a given function with single argument in parallel.

    This function uses multiple cores.
    This function accept single-argument functions.

    Inputs:
        fun: a top-level function taking single argument.
        args: a list of input for single argument
        n_jobs (int): number of cores used. Default = cpus on machine.

    Output:
        A list of function return value with the same length of argument list.
    """
    return Parallel(n_jobs=n_jobs)(delayed(fun)(arg) for arg in args)


def spawn(f):
    def fun(q_in, q_out):
        while True:
            i, x = q_in.get()
            if i is None:
                break
            q_out.put((i, f(x)))
    return fun


def parmap(f, x, nprocs=multiprocessing.cpu_count()):
    """
    Parallel map that can be used with method functions or lambda functions
    contrary to the built multiprocessing map or imap functions.

    This function accepts single-argument functions.

    it leads to error when too much data has to be carried.
    """
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=spawn(f), args=(q_in, q_out))
            for _ in range(nprocs)]

    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(x)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


def parmap_star(f, args, nproces=multiprocessing.cpu_count()):
    """Run parallel map with multiple arguments.

    The function takes multiple arguments.
    The arguments have to to zipped.

    Example:
        def my_fun(a, b):
            return a+b
        a_list = [2, 3, 5]
        b_list = [10, 20, 30]

        >>> parmap_star(my_fun, zip(a_list, b_list))
        [20, 23, 35]
    """
    with Pool(processes=nproces) as pool:
        return pool.starmap(f, args)


def write_h5file(out_path, data_dict, attr_dict=None):
    """Write a dictionary to HDF5.

    Arguments:
        out_path: string
            The output files path.
        data_dict: dictionary
            A dictionary object containing the data.
        attr_dict: dictionary
            A dictionary object containing general attributes.

    Example:
        my_dataset = {}
        my_dataset['voxels'] = np.array([[3,4,5],[6,7,8]])
        my_dataset['labels'] = np.array([0,1])

        write_h5file(my_dataset, './mydataset.h5')
    """
    attr_dict = attr_dict if attr_dict else {}
    with h5py.File(out_path, 'w') as fout:
        # Write datasets
        for key in data_dict:
            data = data_dict[key]
            if isinstance(data, np.ndarray):
                out_type = data.dtype
            else:
                out_type = type(data)
            fout.create_dataset(key, data=data, dtype=out_type)

        # Write attributes
        for key in attr_dict:
            fout.attrs[key] = attr_dict[key]


def read_h5file(file_path):
    """Load data from HDF5 and return a dictionary.

    Arguments:
        file_path: string
            The input HDF5 path.

    Return:
        A dictionary containing the data.
    """
    with h5py.File(file_path, 'r') as fin:
        # keys = attrs if attrs else list(fin.keys())
        data = {}
        for key in fin.keys():
            data[key] = fin[key].value
        for key in fin.attrs:
            data[key] = fin.attrs[key]
        return data


def reshape_to_tf(tensor, dim=3):
    """Reshape the tensor to Tensorflow ordering.

    This function assumes that your patch have
    the same width for each dimension.

    The types of tensors are supported

    - Simple 2D patch sample
      shape = (n_channels, patch_size, patch_size)

    - 2D patch samples corpus
      shape = (n_samples, n_channels, patch_size, patch_size)

    - 3D patch samples corpus
      shape = (n_samples, n_channels, patch_size, patch_size, patch_size)
    """
    if dim == 3:
        n_channels, patch_size, patch_size = tensor.shape
        tensor = tensor.reshape(patch_size, patch_size, n_channels)
    elif dim == 4:
        n_samples, n_channels, patch_size, patch_size = tensor.shape
        tensor = tensor.reshape(n_samples, patch_size, patch_size, n_channels)
    elif dim == 5:
        n_samples, n_channels, ps, ps, ps = tensor.shape
        tensor = tensor.reshape(n_samples, ps, ps, ps, n_channels)
    return tensor


def reshape_to_th(tensor, dim=3):
    """Reshape the tensor to Theano ordering.

    This function assumes that your patch have
    the same width for each dimension.

    The types of tensors are supported

    - Simple 2D patch sample
      shape = (patch_size, patch_size, n_channels)

    - 2D patch samples corpus
      shape = (n_samples, patch_size, patch_size, n_channels)

    - 3D patch samples corpus
      shape = (n_samples, patch_size, patch_size, patch_size, n_channels)
    """
    if dim == 3:
        patch_size, patch_size, n_channels = tensor.shape
        tensor = tensor.reshape(n_channels, patch_size, patch_size)
    elif dim == 4:
        n_samples, patch_size, patch_size, n_channels = tensor.shape
        tensor = tensor.reshape(n_samples, n_channels, patch_size, patch_size)
    elif dim == 5:
        n_samples, ps, ps, ps, n_channels = tensor.shape
        tensor = tensor.reshape(n_samples, n_channels, ps, ps, ps)
    return tensor


def distribute_samples(samples, n_bins):
    """Evenly distrubute the samples into several bins.

    Arguments:
        samples: numpy.ndarray
            All the samples stored in numpy array
        n_bins: int
            Number of splits
    Return:
        A list of numpy array or list
    """
    n_bin_samples = np.zeros((n_bins,), dtype=int)
    n_avg, remain = divmod(len(samples), n_bins)
    n_bin_samples += n_avg

    idx_remain = np.asarray(random.sample(range(n_bins), remain), dtype=int)
    n_bin_samples[idx_remain] += 1

    subsets = []
    idx = 0
    for i in range(n_bins):
        n_bin_sample = n_bin_samples[i]
        subset = samples[idx: idx + n_bin_sample]
        subsets.append(subset)
        idx += n_bin_sample

    return subsets


def change_parent_dir(new_dir, raw_path, new_ext=None):
    """Change the parent directory of given file path.

    Optionally change the extension.

    Example:
        new_dir = '/my/output/'
        raw_path = '/some/where/image.nii'
        new_ext = '.mat'

        >>> change_parent_dir(new_dir, raw_path, new_ext)
        # '/my/output/image.mat'
    """
    filename, ext = os.path.splitext(raw_path)
    filename = os.path.split(filename)[-1]
    if new_ext:
        return os.path.join(new_dir, filename + new_ext)
    else:
        return os.path.join(new_dir, filename + ext)


def run_theano_script(script, gpu, args):
    """Run script with theano config.

    This function sets individual directory for each gpu
    Theano complilation.

    By default, the compile directory is under
    the current directory.

    You also can specify the prefix ojf compile directory
    by setting your environment variable like

    ```
    THEANO_BASE_COMPILEDIR=/some/where/.theano
    ```

    Those compile directories will be deleted after script
    is done.
    """
    # load from environment or set to default directory
    prefix = os.getenv('THEANO_BASE_COMPILEDIR', './.theano')
    base_compiledir = prefix + str(gpu)
    cmd = ("THEANO_FLAGS='device={}, base_compiledir={}' "
           "python {} {}".format(gpu, base_compiledir, script, args))
    os.system(cmd)
    shutil.rmtree(base_compiledir, ignore_errors=True)


def load_mri(mri_path, **kwargs):
    """Load the data from file.

    It could be used to load mri scans or labels.
    The format of label files can be either
    '.mat' or '.nii'

    If the extension is .mat
    one should specify the attribute by something like

        >>> load_mri(my_matfile, key='label')

    Or leave use the default key 'label'
        >>> load_mri(my_matfile)
    """
    ext = os.path.splitext(mri_path)[-1]
    if ext == '.mat':
        tmp = loadmat(mri_path)
        key = kwargs['key'] if 'key' in kwargs else 'label'
        data = tmp[key].squeeze()
    elif ext == '.nii':
        data = nib.load(mri_path).get_data().squeeze()
    return data


def plot_train_history(log_path, fig_path):
    """Plot the train history given log file.

    The log file should contain a header as first row
    epoch, acc, loss, vali_acc, vali_loss

    This function will read and drop the header.
    """
    with open(log_path, 'r') as fin:
        # load log and drop the header
        data = np.array(list(csv.reader(fin, delimiter=',')))[1:]
        epochs = data[:, 0].astype(int)
        train_acc = data[:, 1].astype(np.float32)
        train_loss = data[:, 2].astype(np.float32)
        vali_acc = data[:, 3].astype(np.float32)
        vali_loss = data[:, 4].astype(np.float32)

        # top plot: loss
        plt.suptitle("Model training history")
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_loss, label='train_loss')
        plt.plot(epochs, vali_loss, label='vali_loss')
        plt.legend()
        plt.ylabel("Loss")

        # bottom plot: accuracy
        plt.subplot(2, 1, 2)
        plt.plot(epochs, train_acc, label='train_acc')
        plt.plot(epochs, vali_acc, label='vali_acc')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.savefig(fig_path)


def get_candidate_results(img_path):
    """Get the candidate result given file result path.

    This is a helper function that retreives the
    candidate prediction results in multi-atlas based segmentation method.

    For examples, the path of final prediction results is /seg/my_img.mat

    Then then candidate result files should be under a directory /seg/my_img/
    """
    img_name, ext = os.path.splitext(img_path)
    candidate_pathes = glob.glob(img_name + '/*' + ext)
    candidate_results = []
    for candidate_path in candidate_pathes:
        labels = load_mri(candidate_path)
        labels = convert_to_miccai_labels(labels)
        candidate_results.append(labels)
    return candidate_results


def get_label_count(data, n_labels):
    """Count the label occurence in data.

    Arguments:
        data: list or numpy array
            The data sequence
        n_labels: int
            Number of labels

    Return:
        Occurence of each label. Shape = (n_labels,)
    """
    counts = np.zeros((n_labels,), dtype=int)
    # count the occurence in data
    nz_counts = Counter(data).most_common()
    for k, v in nz_counts:
        counts[k] = v

    return counts


def get_attribute(data_path, key):
    ext = os.path.splitext(data_path)[1]
    if ext == '.mat':
        t = loadmat(data_path)
        return t[key]

# only for MICCAI 2012 multi-label challenge
def load_miccai_labels(label_path):
    """Load miccai manual segmentation."""
    nib.nifti1.Nifti1Header.quaternion_threshold = -1e-6
    lab = load_mri(label_path)
    lab = convert_to_miccai_labels(lab)

    return lab


def convert_to_miccai_labels(lab):
    if lab.max() <= 134:
        return lab
    else:
        # See Miccai rules
        miccai_ignored_labels = list(range(1, 4)) + \
                                list(range(5, 11)) + \
                                list(range(12, 23)) + \
                                list(range(24, 30)) + \
                                [33, 34, 42, 43, 53, 54] + \
                                list(range(63, 69)) + [70, 74] + \
                                list(range(80, 100)) + \
                                [110, 111, 126, 127, 130, 131,
                                 158, 159, 188, 189]

        miccai_true_labels = [4, 11, 23, 30, 31, 32, 35, 36, 37,
                              38, 39, 40, 41, 44, 45, 46, 47, 48,
                              49, 50, 51, 52, 55, 56, 57, 58, 59,
                              60, 61, 62, 69, 71, 72, 73, 75, 76,
                              100, 101, 102, 103, 104, 105, 106,
                              107, 108, 109, 112, 113, 114, 115,
                              116, 117, 118, 119, 120, 121, 122,
                              123, 124, 125, 128, 129, 132, 133,
                              134, 135, 136, 137, 138, 139, 140,
                              141, 142, 143, 144, 145, 146, 147,
                              148, 149, 150, 151, 152, 153, 154,
                              155, 156, 157, 160, 161, 162, 163,
                              164, 165, 166, 167, 168, 169, 170,
                              171, 172, 173, 174, 175, 176, 177,
                              178, 179, 180, 181, 182, 183, 184,
                              185, 186, 187, 190, 191, 192, 193,
                              194, 195, 196, 197, 198, 199, 200,
                              201, 202, 203, 204, 205, 206, 207]
        return label_filtering(lab, miccai_ignored_labels, miccai_true_labels)
