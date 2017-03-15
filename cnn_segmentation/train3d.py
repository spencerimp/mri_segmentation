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
    crop_images,
    get_coordinates,
    image_smooth,
    load_miccai_labels,
    pad_images,
    parmap_star,
    read_h5file,
    reshape_to_tf,
    shuffle_all,
    write_h5file,
    zscore,
)
from utils.voxel_sampling import (
    PickVoxelBalanced,
)
from utils.voxel_feature import (
    Pick3DPatchMultiScale,
)
from network import (
    Cnn3DPatch,
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
                         patch_size,
                         scales,
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
        n_voxels: int
            Number of voxels needed in the image

    Return:
        patches: feature array. Shape = (n_voxels, 6, patch_size, patch_size)
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
    voxels = voxel_picker.pick_voxels(n_voxels, expand_boundary=True)

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
    # extract 6patch feature
    patch_picker_class = Pick3DPatchMultiScale
    # set the shared arguments
    partial_extract_image = partial(extract_single_image,
                                    voxel_picker_class=voxel_picker_class,
                                    patch_picker_class=patch_picker_class,
                                    patch_size=patch_size,
                                    scales=scales,
                                    n_voxels=n_voxels_tr+n_voxels_va)
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
        all_patches = reshape_to_tf(all_patches, dim=5)
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


def extract_save_subsets(img_pathes, lab_pathes,
                         train_pathes, n_voxels_tr,
                         n_voxels_va,
                         patch_size, scales):
    """Extract validation and training and save to disk.

    Collect validation voxels from all the images and save them
    as a fix validation set. Such validation is regarded as
    "general unseen data". There might be redundance within
    the validation set, and there might be overlap between training
    and validation sets because we allow sampling with repetition.

    Sequentially process the training images and save them
    as separated subsets to disk.

    Return:
        validation feature, validation label
    """
    # sample validations only (small enough to afford parallism)
    print("Start to extract validation set")
    _, _, x_va, y_va = extract_data(img_pathes, lab_pathes,
                                    0, n_voxels_va,
                                    patch_size, scales,
                                    is_parallel=True)

    # sample and save training subsets sequentially
    print("Start to extract training set")
    for img_path, lab_path, train_path in zip(img_pathes,
                                              lab_pathes,
                                              train_pathes):
        # one image for one output
        if not os.path.exists(train_path):
            extract_data([img_path], [lab_path],
                         n_voxels_tr, 0,
                         patch_size, scales,
                         is_parallel=False,
                         out_path=train_path)
    return x_va, y_va


def build_network(model_path, logger_path,
                  stats_path, n_classes, patch_size, scales,
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
        patch_size: string
            The input patch size
        retrain: bool
            Whether to train the weights from scratch
        optimizer: Keras Optimizer instance
            The optimizer used in training
            Default: Adagrade(lr=0.01, epsilon=1e-08, decay=0.0)
        init: string
            The weight initilization method
            Default: 'glorot_uniform'
    """
    n_channels = len(scales)
    net = Cnn3DPatch(patch_size=patch_size,
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
def train3DPatch(img_pathes, lab_pathes,
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
        is_parallel: bool, Default = True
            Whether to extract feature from multiple image in parallel
        retrain: bool, Default = True
            Whether to train the weights from scratch
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


def train3DPatch_generator(img_pathes, lab_pathes,
                           model_path, stats_path, logger_path,
                           n_voxels_tr, n_voxels_va,
                           n_classes, patch_size, scales,
                           batch_size=200, max_epoch=300,
                           optimizer=None, init='glorot_uniform',
                           is_parallel=True, retrain=True):
    """Train a CNN with two sets of triplanar patches using generator.

    Arguments:
        img_pathes: list of string
            The path of images
        lab_pathes: list of string
            The path of corresponding manual label
        model_path: string
            The output path network model
        stats_path: string
            The output of network statistics
        n_voxels_tr: int
            Number of training from each training image
        n_voxels_va: int
            Number of validation voxels from each training image
        patch_size: int
            The size of input patch
        batch_size: int
            The batch size when performing stochastic gradient descent
            Default = 200
            This is also the number of samples the generator yields each time
        max_epoch: int
            The max number of epoch to train. Default = 300
        optimizer: keras optimizer instance
            The optimizer of training.
            Default: Adam
        init: string
            The scheme of parameter initilization
            Default: 'glorot_uniform'
        is_parallel: bool, Default = True
            Whether to extract feature from multiple image in parallel
        retrain: bool, Default = True
            Whether to train the weights from scratch
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

    # load all the feature into memory at once
    def feat_gen(x_tr, y_tr, batch_size):
        # run until nb_epoch has reached max
        idx = 0
        while True:
            # yields batches to train in this epoch
            batch_slice = slice(idx, idx + batch_size)
            yield x_tr[batch_slice], y_tr[batch_slice]
            idx += batch_size

    print("Normalizing data")
    x_tr, mu, sigma = zscore(x_tr)
    x_va = zscore(x_va, mu=mu, sigma=sigma)[0]
    y_tr = np_utils.to_categorical(y_tr, net.out_size)
    y_va = np_utils.to_categorical(y_va, net.out_size)

    gen = feat_gen(x_tr, y_tr, batch_size)
    net.set_train_stats(mu, sigma, len(y_tr), len(y_va))
    print("Start to train the model")
    net.train_generator(gen, len(y_tr),
                        x_va, y_va,
                        max_epoch=max_epoch)


def train3DPatch_online(img_pathes, lab_pathes,
                        model_path, stats_path, logger_path,
                        n_voxels_tr, train_path_template,
                        n_voxels_va,
                        n_classes, patch_size, scales,
                        batch_size=200, max_epoch=300,
                        optimizer=None, init='glorot_uniform'):
    """Train a CNN with two sets of triplanar patches in online fashion.

    DO NOT USE IT NOW.

    Collect validation voxels from all the images and save them
    as a fix validation set. Such validation is regarded as
    "general unseen data". There might be redundance within
    the validation set, and there might be overlap between training
    and validation sets because we allow sampling with repetition.

    Sequentially process the training images, shuffle and save them
    as separated subsets to disk.
    For each epoch during training phrase, load all the
    subsets with Python generator with small batches.

    This function is called online as it does not shuffle the voxels
    across different training images; it trains the model by gradually
    loading different new training images one after another rather than
    seeing all of them at once.

    All the datasets are normzalized by the feature in first training image.

    Arguments:
        img_pathes: list of string
            The path of images
        lab_pathes: list of string
            The path of corresponding manual label
        model_path: string
            The output path network model
        stats_path: string
            The output of network statistics
        logger_path: string
            The output of training log
        n_voxels_tr: int
            Number of training from each training image
        train_path_template: string
            The path prefix of trainining set (e.g. /home/mydata/train.h5)
        n_voxels_va: int
            Number of validation voxels from each training image
        patch_size: int
            The size of input patch
        batch_size: int
            The batch size when performing stochastic gradient descent
            This should be small. Default = 200
            This is also the generator size
        max_epoch: int
            The max number of epoch to train. Default = 300
        optimizer: keras optimizer instance
            The optimizer of training. Default: Adam
        init: string
            The scheme of parameter initilization
    """
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))

    # save training sets to disk and get validation
    t_start = time.time()
    prefix, ext = os.path.splitext(train_path_template)
    train_pathes = []
    for img_path in img_pathes:
        img_file = os.path.splitext(img_path)[0]
        img_name = os.path.split(img_file)[-1]
        train_pathes.append(os.path.join(prefix+img_name+ext))

    x_va, y_va = extract_save_subsets(img_pathes, lab_pathes,
                                      train_pathes, n_voxels_tr,
                                      n_voxels_va,
                                      patch_size, scales)

    print("Feature extracted in {} seconds".format(time.time()-t_start))
    net = build_network(model_path, logger_path,
                        n_classes, patch_size, scales,
                        optimizer, init)

    # normalization
    ds_train = read_h5file(train_pathes[0])
    x_tr = ds_train['feat']
    y_tr = ds_train['label']
    _, mu, sigma = zscore(x_tr)

    # load the trainin subsets sequentially with generator
    n_all_voxels = len(img_pathes) * n_voxels_tr

    def online_generator(train_pathes, mu, sigma, batch_size):
        while True:
            for train_path in train_pathes:
                # Note that the Keras progress bar will override some lines
                print("    Load data from {}".format(train_path))
                ds_train = read_h5file(train_path)
                x_tr = ds_train['feat']
                y_tr = ds_train['label']
                y_tr = np_utils.to_categorical(y_tr, 135)
                # yields batches to train in this epoch
                idx = 0
                n_batches, m = divmod(len(y_tr), batch_size)
                if m > 0:
                    n_batches += 1
                for i in range(n_batches):
                    batch_slice = slice(idx, idx + batch_size)
                    x_batch = x_tr[batch_slice]
                    x_batch = zscore(x_batch, mu=mu, sigama=sigma)[0]
                    y_batch = y_tr[batch_slice]
                    yield x_batch, y_batch
                    idx += batch_size

    print("Normalizing data")
    y_va = np_utils.to_categorical(y_va, net.out_size)
    x_va = zscore(x_va, mu=mu, sigma=sigma)[0]
    gen = online_generator(train_pathes, mu, sigma, batch_size)
    net.set_train_stats(mu, sigma, len(y_tr), len(y_va))
    print("Start to train the model")
    net.train_generator(gen, n_all_voxels,
                        x_va, y_va, max_epoch=max_epoch)

    # delete the datsets to save space
    for train_path in train_pathes:
        os.remove(train_path)


if __name__ == '__main__':
    # set parameters
    nib.nifti1.Nifti1Header.quaternion_threshold = -1e-6
    img_dir = './datasets/miccai/train/mri/'
    lab_dir = './datasets/miccai/train/label/'
    model_path = './experiments/testtt/cnn_6patches.h5'
    stats_path = './experiments/testtt/cnn_6patches_stat.h5'
    logger_path = './experiments/testtt/train6patches_log.csv'
    n_classes = 135
    patch_size = 13
    scales = [1]
    n_voxels_tr = 40000
    n_voxels_va = 10000
    batch_size = 200
    max_epoch = 300
    extract_parallel = True

    # parse and run
    img_pathes = glob.glob(img_dir+'*.nii')
    lab_pathes = [change_parent_dir(lab_dir, img_path, '_glm.nii')
                  for img_path in img_pathes]
    learning_rate = 0.05
    momentum = 0.5
    optimizer = SGD(learning_rate, momentum)
    # optimizer = None #  use default optimizer
    train3DPatch(img_pathes, lab_pathes,
                 model_path, stats_path, logger_path,
                 n_classes, patch_size, scales,
                 n_voxels_tr, n_voxels_va,
                 batch_size, max_epoch,
                 optimizer, extract_parallel)
