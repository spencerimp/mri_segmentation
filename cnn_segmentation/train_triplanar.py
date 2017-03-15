import os
import glob
import time
import numpy as np
import nibabel as nib
from itertools import starmap
from functools import partial
from utils.utils import (
    change_parent_dir,
    convert_voxels_padding,
    get_coordinates,
    image_smooth,
    load_miccai_labels,
    pad_images,
    parmap_star,
    reshape_to_tf,
    shuffle_all,
    write_h5file,
    zscore,
)
from utils.voxel_sampling import (
    PickVoxelBalanced,
)
from utils.voxel_feature import (
    PickTriplanarMultiPatch,
)
from network import (
    CnnTriplanarMultiset,
)
from keras.optimizers import (
    SGD,
)
from keras.utils import np_utils
import keras.backend as K


# Inner functions
def extract_single_image(img_path, lab_path,
                         voxel_picker_class,
                         patch_picker_class,
                         patch_size, scales,
                         n_voxels):
    """Sample and extract feature from an image.

    Arguments:
        img_path: string
            The image path
        lab_path: string
            The corresponding label path
        voxel_picker_class: string
            The name of voxels picker class (e.g. PickVoxelBalanced)
        patch_picker_class: string
            The name of patch picker class (e.g. PickTriplanarMultiPatch)
        patch_size: int
            The input patch size
        scales: list of int
            The scales used
        n_voxels: int
            Number of voxels needed in the image

    Return:
        patches: feature array. Shape = (n_voxels, 3*len(scales), patch_size, patch_size)
        patch_labels: label array. Shape = (n_voxels,)
    """
    print("Load training image {}".format(img_path))
    img = nib.load(img_path).get_data().squeeze()
    img = img.astype(np.float32)
    lab = load_miccai_labels(lab_path)
    lab = lab.astype(int)

    img = image_smooth(img, lab)

    # extract voxels
    all_voxels = get_coordinates(lab.shape)
    voxel_picker = voxel_picker_class(lab, all_voxels)
    voxels = voxel_picker.pick_voxels(n_voxels)

    # extract patches
    pad_width = int((patch_size * np.max(scales) - 1) / 2)
    img, lab = pad_images(pad_width, img, lab)
    voxels = convert_voxels_padding(voxels, pad_width)

    patch_picker = patch_picker_class(img, patch_size, scales)
    patches = patch_picker.pick(voxels)
    patch_labels = lab[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    return patches, patch_labels


def extract_data(img_pathes, lab_pathes,
                 n_voxels_tr, n_voxels_va,
                 patch_size, scales,
                 is_parallel=False, out_path=None):
    """Extract dataset with several images.

    Arguments:
        img_pathes: list of string
            List of image pathes
        lab_pathes: list of string
            List of corresponding label
        n_voxels_tr: int
            Number of training voxels
        n_voxels_va: int
            Number of validation voxels
        patch_size: string
            The input patch size
        scales: list of int
            The scales used
        is_parallel: bool
            Whether to extract images in parallel
            Default = False
        out_path: string
            Write the training and validation into disk
            Leave it empty to skip.

    Return:
        x_tr: feature array of training set
        y_tr: label vector of training set
        x_va: feature array of validation set
        y_va: label vector of validation set
    """
    # sample evenly among the classes
    voxel_picker_class = PickVoxelBalanced
    # extract triplanar patch feature
    patch_picker_class = PickTriplanarMultiPatch
    # set the shared arguments
    partial_extract_image = partial(extract_single_image,
                                    voxel_picker_class=voxel_picker_class,
                                    patch_picker_class=patch_picker_class,
                                    patch_size=patch_size,
                                    scales=scales,
                                    n_voxels=n_voxels_tr+n_voxels_va)
    # extract data from multiple images
    if is_parallel:
        all_results = parmap_star(partial_extract_image,
                                  zip(img_pathes, lab_pathes))
    else:
        all_results = starmap(partial_extract_image,
                              zip(img_pathes, lab_pathes))

    # unwrap the (patches, labels) tuples, and concatenate arrays
    all_patches, all_labels = zip(*all_results)
    all_patches = np.concatenate(all_patches)
    all_labels = np.asarray(all_labels, int)
    all_labels = np.concatenate(all_labels)

    # shuffle before split
    all_patches, all_labels = shuffle_all(all_patches, all_labels)

    if K.image_dim_ordering() == 'tf':
        all_patches = reshape_to_tf(all_patches, dim=4)

    n_all_tr_voxels = n_voxels_tr * len(img_pathes)
    x_tr = all_patches[:n_all_tr_voxels]
    y_tr = all_labels[:n_all_tr_voxels]
    x_va = all_patches[n_all_tr_voxels:]
    y_va = all_labels[n_all_tr_voxels:]

    # does not override
    if out_path:
        dataset = {}
        dataset['feat'] = all_patches
        dataset['label'] = all_labels
        write_h5file(out_path, dataset)
    return x_tr, y_tr, x_va, y_va


def build_network(model_path, logger_path, stats_path,
                  n_classes, patch_size, scales,
                  retrain, **kwargs):
    """
    Extract feature and build the network.
    This should be called before training the network

    Arguments:
        model_path: string
            The path of network model
        logger_path: string
            The path of training logger file
        stats_path: string
            The path of training set statistics file
        n_classes: int
            Number of classes including background
        patch_size: string
            The input patch size
        scales: list of int
            The scales used
        retrain: bool
            Whether to train network from scratch
            Default: True
        optimizer: Keras Optimizer instance
            The optimizer used in training
            Default: Adagrade(lr=0.01, epsilon=1e-08, decay=0.0)
        init: string
            The weight initilization method
            Default: 'glorot_uniform'
    """
    n_channels = 3 * len(scales)
    net = CnnTriplanarMultiset(patch_size=patch_size,
                               n_channels=n_channels,
                               out_size=n_classes)
    # specify stats
    net.model_path = model_path
    net.logger_path = logger_path
    net.stats_path = stats_path
    net.init = kwargs['init']
    net.optimizor = kwargs['optimizer']
    net.build()
    if retrain:
        print("Building a network")
        net.compile(**kwargs)
    else:
        print("Load pre-trained network")
        net.load_model(model_path)
    net.model.summary()
    return net


# APIs
def train_triplanar(img_pathes, lab_pathes,
                    model_path, stats_path, logger_path,
                    n_classes, patch_size, scales,
                    n_voxels_tr, n_voxels_va,
                    batch_size=200, max_epoch=300,
                    optimizer=None, init='glorot_uniform',
                    is_parallel=True, retrain=True):
    """Train a CNN with two sets of triplanar patches.

    Arguments:
        img_pathes: list of string
            The path of images
        lab_pathes: list of string
            The path of corresponding manual label
        model_path: string
            The output path network model
        stats_path: string
            The output of network statistics
        patch_size: int
            The size of input patch
        scales: list of int
            The scales used
        n_voxels_tr: int
            Number of training from each training image
        n_voxels_va: int
            Number of validation voxels from each training image
        batch_size: int
            The batch size when performing stochastic gradient descent
            This should be small. Default = 200
        max_epoch: int
            The max number of epoch to train. Default = 300
        optimizer: keras optimizer instance
            The optimizer of training. Default: Adam
        is_parallel: bool
            Whether to extract feature from multiple image in parallel
            Defulat: True
        retrain: bool
            Whether to train network from scratch
            Default: True
    """
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))

    t_start = time.time()
    x_tr, y_tr, x_va, y_va = extract_data(img_pathes, lab_pathes,
                                          n_voxels_tr, n_voxels_va,
                                          patch_size, scales,
                                          is_parallel)

    print("Feature extracted in {} seconds".format(time.time()-t_start))
    net = build_network(model_path, logger_path,
                        stats_path, n_classes, patch_size, scales,
                        retrain,
                        optimizer=optimizer,
                        init=init)

    print("Normalizing data")
    x_tr, mu, sigma = zscore(x_tr)
    x_va = zscore(x_va, mu=mu, sigma=sigma)[0]
    y_tr = np_utils.to_categorical(y_tr, net.out_size)
    y_va = np_utils.to_categorical(y_va, net.out_size)
    net.set_train_stats(mu, sigma, len(y_tr), len(y_va))
    print("Start to train the model")
    net.train(x_tr, y_tr,
              x_va, y_va,
              batch_size=batch_size,
              max_epoch=max_epoch)


if __name__ == '__main__':
    # set parameters
    nib.nifti1.Nifti1Header.quaternion_threshold = -1e-6
    img_dir = './datasets/miccai/train/mri/'
    lab_dir = './datasets/miccai/train/label/'
    model_path = './experiments/testtt/cnn_6patches.h5'
    stats_path = './experiments/testtt/cnn_6patches_stat.h5'
    logger_path = './experiments/testtt/train6patches_log.csv'
    patch_size = 29
    scales = [1, 3]
    n_voxels_tr = 40000
    n_voxels_va = 10000
    batch_size = 200
    max_epoch = 300
    init = 'he_uniform'
    extract_parallel = True
    retrain = True

    # parse and run
    img_pathes = glob.glob(img_dir+'*.nii')
    lab_pathes = [change_parent_dir(lab_dir, img_path, '_glm.nii')
                  for img_path in img_pathes]
    learning_rate = 0.05
    momentum = 0.5
    optimizer = SGD(learning_rate, momentum)
    train_triplanar(img_pathes, lab_pathes,
                    model_path, stats_path, logger_path,
                    patch_size, scales, n_voxels_tr, n_voxels_va,
                    batch_size, max_epoch,
                    optimizer, init, extract_parallel, retrain)
