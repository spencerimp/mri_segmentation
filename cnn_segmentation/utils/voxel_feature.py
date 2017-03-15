from __future__ import print_function
import numpy as np
import sys
sys.path.append('../')
from utils.utils import get_label_count


def get_downscaled_tensor(raw_tensor, scale):
    """
    Take a tensor, downscale by taking the average among the [scale] points.

    Inputs:
        raw_tensor (numpy array-like): raw array or matrix
        scale (int): the stride to take the mean pooing (for all axis).

    Output:
        Shrinked tensor, which shape = raw_tensor.shape/scale

    Example:
        >>> import numpy as np
        >>> raw_tensor = np.array([[1,2,3],
                                  [4,5,6],
                                  [7,8,9]])
        >>> scale = 3
        # downscale a 3-by-3 matrix by 3, getting 1-by-1
        >>> get_downscaled_tensor(raw_tensor, 3)
        # should be a 1-by-1 matrix [[5]] (average of the average of each row)
    """
    # do nothing if the user does not want to downscales it
    if scale == 1:
        return raw_tensor

    # reshape the data so that each group in stored in one axis
    avg_shape = np.asarray(raw_tensor.shape) // scale

    shape_list = []
    for s in avg_shape:
        shape_list.extend([s, scale])
    avg_tensor = raw_tensor.reshape(shape_list)

    # take mean pooing iteratively for each axis to shrink the tensor
    for i in range(len(avg_shape)):
        avg_tensor = avg_tensor.mean(i + 1)

    return avg_tensor


class PickFeature():
    """
    Abstract class for the feature extraction of the center voxels in an image.
    Each feature will be vectorized as an array.
    """
    def pick(self, vxs):
        pass

    def pick_generator(self, vxs, batch_size=100000):
        n_voxels = vxs.shape[0]
        n_batches, n_remain = divmod(n_voxels, batch_size)
        if n_remain > 0:
            n_batches += 1
        idx = 0
        for _ in range(n_batches):
            vx_batch = vxs[idx: idx + batch_size]
            patches = self.pick(vx_batch)
            yield patches


class PickPatchFeature(PickFeature):
    """
    Abstract class for patch-based features.
    """
    def __init__(self, img, patch_shape):
        self.img = img
        self.patch_shape = patch_shape


class Pick2DPatch(PickPatchFeature):
    """
    Pick up single 2D square slice.
    """
    def __init__(self, img, patch_size, orth_axis=0, downscale=1):
        self.patch_size = patch_size
        self.orth_axis = orth_axis
        self.downscale = downscale
        self.slice_axis = list(range(3))  # x, y, z
        del self.slice_axis[orth_axis]
        out_patch_size = int(patch_size / float(downscale))
        patch_shape = [out_patch_size] * 2
        PickPatchFeature.__init__(self, img, patch_shape)

    def pick(self, vxs):
        """Extract a 2D slice.

        output: 4d tensor,
                shape = (n_voxels, 1, patch_size, patch_size).
        """
        slice3d = [slice(None)] * 3
        n_voxels = vxs.shape[0]

        # take ceil to deal with even patch size
        radius = int(np.ceil((self.patch_size - 1) / 2))

        # extract slice and vectorize
        out_shape = tuple([n_voxels, 1] + self.patch_shape)
        patches = np.zeros(out_shape, dtype=self.img.dtype)
        for i, vx in enumerate(vxs):
            # get the whole plane at a specific point
            slice3d[self.orth_axis] = vx[self.orth_axis]
            img2d = self.img[slice3d]
            vx2d = vx[self.slice_axis]
            # retrieve the needed patch, centered at the voxel
            if self.patch_size % 2 == 1:
                patch = img2d[vx2d[0] - radius:vx2d[0] + radius + 1,
                              vx2d[1] - radius:vx2d[1] + radius + 1]
            else:
                # if the patch size is not odd, count on left
                patch = img2d[vx2d[0] - radius:vx2d[0] + radius,
                              vx2d[1] - radius:vx2d[1] + radius]
            # downscale by taking average
            patch = get_downscaled_tensor(patch, self.downscale)
            patches[i] = patch
        return patches


class PickTriplanarPatch(PickPatchFeature):
    """
    Pick up three orthogonal 2D square slices.
    """
    def __init__(self, img, patch_size, downscale=1):
        self.patch_size = patch_size
        self.pick2d_list = [Pick2DPatch(img, patch_size,
                                        axis, downscale)
                            for axis in range(3)]
        out_patch_size = int(patch_size / float(downscale))
        patch_shape = [out_patch_size] * 2
        PickPatchFeature.__init__(self, img, patch_shape)

    def pick(self, vxs):
        """Extract 2D slices and put into different channels.

        output: 4d tensor,
                shape = (n_voxels, 3, patch_size, patch_size)
        """
        n_voxels = vxs.shape[0]
        out_shape = tuple([n_voxels, 3] + self.patch_shape)
        patches = np.zeros(out_shape, dtype=self.img.dtype)
        for axis, pick2d in enumerate(self.pick2d_list):
            patch = pick2d.pick(vxs)
            patches[:, axis, :, :] = patch.squeeze()
        return patches


class PickTriplanarMultiPatch(PickPatchFeature):
    """
    Pick up several of triplanar patches.
    """
    def __init__(self, img, patch_size, scales):
        self.patch_size = patch_size
        self.scales = scales
        self.pick2d_list = []
        for scale in scales:
            pickers = [Pick2DPatch(img, patch_size * scale,
                                   axis, scale)
                       for axis in range(3)]

            self.pick2d_list.extend(pickers)
        out_patch_size = patch_size
        patch_shape = [out_patch_size] * 2
        PickPatchFeature.__init__(self, img, patch_shape)

    def pick(self, vxs):
        n_voxels = vxs.shape[0]
        out_shape = tuple([n_voxels, 3 * len(self.scales)] + self.patch_shape)
        patches = np.zeros(out_shape, dtype=self.img.dtype)
        for axis, pick2d in enumerate(self.pick2d_list):
            patch = pick2d.pick(vxs)
            patches[:, axis, :, :] = patch.squeeze()
        return patches


class Pick3DPatch(PickPatchFeature):
    """
    Pick up a 3D cube.
    """
    def __init__(self, img, patch_size, downscale):
        self.patch_size = patch_size
        self.downscale = downscale
        out_patch_size = int(patch_size / float(downscale))
        patch_shape = [out_patch_size] * 3
        PickPatchFeature.__init__(self, img, patch_shape)

    def pick(self, vxs):
        """Extract a 3D cube.

        output: 5d tensor,
                shape = (n_voxel, 1, patch_size, patch_size, patch_size)
        """
        n_voxels = vxs.shape[0]
        radius = int((self.patch_size - 1) / 2)

        # out_shape = (n_voxels, 1, patch_size, patch_size, patch_size)
        out_shape = tuple([n_voxels, 1] + self.patch_shape)
        patches = np.zeros(out_shape, dtype=self.img.dtype)
        for i, vx in enumerate(vxs):
            patch = self.img[vx[0] - radius: vx[0] + radius + 1,
                             vx[1] - radius: vx[1] + radius + 1,
                             vx[2] - radius: vx[2] + radius + 1]

            # downscale
            patch = get_downscaled_tensor(patch, self.downscale)
            patches[i] = patch

        return patches


class Pick3DPatchMultiScale(PickPatchFeature):
    """
    Pick up a 3D cube.
    """
    def __init__(self, img, patch_size, scales):
        self.patch_size = patch_size
        self.scales = scales
        self.pick3d_list = []
        for scale in scales:
            pickers = [Pick3DPatch(img, patch_size * scale,
                                   scale)]

            self.pick3d_list.extend(pickers)
        out_patch_size = patch_size
        patch_shape = [out_patch_size] * 3
        PickPatchFeature.__init__(self, img, patch_shape)

    def pick(self, vxs):
        n_voxels = vxs.shape[0]
        out_shape = tuple([n_voxels, len(self.scales)] + self.patch_shape)
        patches = np.zeros(out_shape, dtype=self.img.dtype)
        for axis, pick3d in enumerate(self.pick3d_list):
            patch = pick3d.pick(vxs)
            patches[:, axis, :, :] = patch.squeeze()
        return patches


class PickDenseFeature(PickFeature):
    """"Pick up feature for dense layer.

    In contrary to input for 2D or 3D,
    this feature is 1D and used (mainly) for the input
    of dense (fully-connceted) layer.
    """
    def __init__(self, out_dim):
        self.out_dim = out_dim


class PickDistCentroids(PickDenseFeature):
    """
    Pick up distance to centroids vector.
    """
    def __init__(self, lab, regions):
        self.out_dim = len(regions)
        self.lab = lab
        self.regions = regions
        self.centroids = np.zeros((len(regions), 3))
        for i, reg_id in enumerate(self.regions):
            idx = np.where(lab == reg_id)
            vxs = np.array(list(zip(idx[0], idx[1], idx[2])))
            self.centroids[i] = np.mean(vxs, axis=0)

    def pick(self, vxs):
        """Extract the distance to centroids.

        output: (n_voxel, n_regions)
            The Euclidean distance from the voxel to
            the centroid of each region.
        """
        dist_cents = [np.linalg.norm(self.centroids - vx, axis=1)
                      for vx in vxs]

        return np.asarray(dist_cents)


class PickRegProb(PickDenseFeature):
    """
    Pick up probability from multi-atlas registration results.

    Note that the probability of label zero is ignored.
    """
    def __init__(self, n_labels, candidate_results):
        self.n_labels = n_labels
        self.candidate_results = candidate_results

    def pick(self, vxs):
        """Extract probability estimates from multi-atlas results."""
        n_voxels = len(vxs)
        probs = np.zeros((n_voxels, self.n_labels))
        for i, (x, y, z) in enumerate(vxs):
            votes = [res[x, y, z] for res in self.candidate_results]
            counts = get_label_count(votes, self.n_labels)
            probs[i] = counts / len(self.candidate_results)

        return probs


class PickExtProb(PickDenseFeature):
    """
    Pick up probability estimate of each class.
    """
    def __init__(self, n_labels, prob_all):
        self.n_labels = n_labels
        self.prob_all = prob_all

    def pick(self, vxs):
        """Extract probability estimates from multi-atlas results."""
        return self.prob_all[vxs[:, 0], vxs[:, 1], vxs[:, 2]]
