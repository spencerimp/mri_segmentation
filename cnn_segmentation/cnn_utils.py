from __future__ import print_function
import os
import glob
import numpy as np
from scipy.io import (
    loadmat,
    savemat,
)
from utils.utils import (
    convert_voxels_padding,
    convert_voxels_original,
    get_candidate_results,
    get_coordinates,
    get_nonzero_voxels,
    image_smooth,
    load_mri,
    load_miccai_labels,
    pad_images,
    recover_image,
    reshape_to_tf,
    zscore,
)
from utils.voxel_feature import (
    PickPatchFeature,
    PickDistCentroids,
)
from utils.voxel_sampling import (
    PickVoxelBalancedNonBackground,
)


def extract_feat_generator(net, voxels, batch_size,
                           feat_pickers, mus, sigmas):
    """Pick voxel features with generator.

    This function extracts multiple voxels features given
    the input feature pickers and returns a tuple of
    features.

    This could be used for single-input or multi-input model.

    Arguments:
        net: Network
            The trained network
        voxels: numpy.ndarray, shape = (n_voxels, 3)
            The voxels to be extracted (on cropped and padded domain)
        batch_size: int
            The size of extracted batch.
            Could be larger than the GPU prediction batch size.
        feat_pickers: list of PickPatchFeature or PickDenseFeature instance
            The feat_picker should have a method pick(vxs) to pick feature of
            voxels.

            Each feature picker extracts one voxel feature from a single image.

    Yields:
        Tuple of voxel features in batch
        If only one feature picker is set. The tuple will be dropped.
    """
    # split into batches
    n_batches, rem = divmod(voxels.shape[0], batch_size)
    if rem > 0:
        n_batches += 1
    idx = 0
    for i in range(n_batches):
        print("Processing feature batch {}/{}".format(i + 1, n_batches))
        vxs = voxels[idx:idx + batch_size]

        # container for all kinds of features
        all_feat = []

        # set up individual feature pickers
        for feat_picker, mu, sigma in zip(feat_pickers, mus, sigmas):
            feat = feat_picker.pick(vxs)

            # we only adjust the patch features (2D or 3D)
            if isinstance(feat_picker, PickPatchFeature):
                if net.image_dim_ordering == 'tf':
                    feat = reshape_to_tf(feat)

            # normalize the test data here
            feat = zscore(feat, mu=mu, sigma=sigma)[0]
            all_feat.append(feat)
        idx += batch_size

        # Drop the unnecessary brackets for single-input model
        if len(all_feat) == 1:
            # yield feat0
            yield all_feat[0]
        else:
            # yield [feat0, feat1,..., featN]
            yield all_feat


def segment_by_generator(net, img_path, mask_path,
                         patch_size, scales, patch_picker_class,
                         batch_size, gpu_batch_size, **kwargs):
    """Segment an unseen image.

    Segment an unseems image using a trained network.
    The mask is used to prune unrelavant voxels.

    Arguments:
        net: trained network
        img_path: string
            The path of mri image
        mask_path: string
            The path of mask
        patch_size: int
            The patch size of input
        scales: list of int
            The scales used
        pad_width: int
            The padding width of image
        batch_size: int
            The size of each feature batch
        gpu_batch_size: int
            The size of each batch sent to GPU prediciton
    """
    print("Segmenting {}".format(img_path))
    img = load_mri(img_path)
    img = img.astype(np.float32)
    temp = loadmat(mask_path)
    mask = temp['label']
    img_shape = img.shape
    img = image_smooth(img, mask) #  only for miccai

    # extract voxels
    all_voxels = get_coordinates(mask.shape)
    voxel_picker = PickVoxelBalancedNonBackground(mask, all_voxels)
    voxels = voxel_picker.pick_all()
    n_voxels = voxels.shape[0]
    print("{} voxels to be predicted".format(n_voxels))
    orig_voxels = voxels

    # extract patches
    pad_width = int((patch_size * np.max(scales) - 1) / 2)
    img, mask = pad_images(pad_width, img, mask)
    voxels = convert_voxels_padding(voxels, pad_width)

    network_type = net.network_type
    if network_type == 'CnnTriplanarMultiset':
        patch_picker = patch_picker_class(img, patch_size, scales)
        # get training data statistics
        patch2d_mu = net.patch2d_mu
        patch2d_sigma = net.patch2d_sigma

        feat_gen = extract_feat_generator(net, voxels, batch_size,
                                          [patch_picker],
                                          [patch2d_mu],
                                          [patch2d_sigma])

    elif network_type == 'CnnTriplanarMultisetCentroids':
        patch_picker = patch_picker_class(img, patch_size, scales)
        cent_picker = PickDistCentroids(mask, kwargs['regions'])
        patch2d_mu = net.patch2d_mu
        patch2d_sigma = net.patch2d_sigma
        cent_mu = net.dense_mu
        cent_sigma = net.dense_sigma

        feat_gen = extract_feat_generator(net, voxels, batch_size,
                                          [patch_picker, cent_picker],
                                          [patch2d_mu, cent_mu],
                                          [patch2d_sigma, cent_sigma])

    y_prob = net.predict_generator(feat_gen, gpu_batch_size)
    y_pred = np.argmax(y_prob, axis=1)
    y_pred = recover_image(orig_voxels, y_pred, img_shape)
    if 'prob_path' in kwargs:
        prob_path = kwargs['prob_path']
        data = {}
        data['voxel'] = orig_voxels
        data['prob'] = y_prob
        savemat(kwargs['prob_path'], data)

    return y_pred
