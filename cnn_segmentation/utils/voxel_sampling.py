from __future__ import print_function
import numpy as np
from skimage.morphology import cube, dilation


def create_boundary(lab, regions, width):
    """Create boundary of each region.

    For each non-zeros regions, create a
    boundary of dilation. Take the union
    of these boundary areas to have a
    so-called boundary zone, and assign it
    a new label as max(regions) + 1

    Omit the new boundary voxel if it overlaps
    with any non-zero region.

    For example, the input labeling has non-background regions
    [1, 2, 3], then the corresponding boundary regions
    are [4, 5, 6].

    Arguments:
        lab: numpy array
            The 3d labeling matrix
        regions: list or array of int
            The non-background region list
        width: int
            The boundary width
    """
    kernel = cube(2 * width + 1)
    lab_dilated = lab.copy()
    n_regions = len(regions)
    idx_protected = np.in1d(lab.ravel(),
                            regions).reshape(lab.shape)
    for region in regions:
        lab_binary = np.zeros(lab.shape, dtype=lab.dtype)
        lab_binary[np.where(lab == region)] = 1
        lab_boundary = dilation(lab_binary, kernel) - lab_binary

        # assign a label to this boundary
        idx_boundary = (lab_boundary == 1)
        lab_dilated[idx_boundary & ~idx_protected] = region + n_regions
    return lab_dilated


class PickVoxel():
    """
    Template of picking voxels.
    """
    def __init__(self, labels, voxels, ignored_labels=None):
        assert np.prod(labels.shape) == voxels.shape[0]
        self.labels = labels
        self.regions = np.unique(labels)
        self.voxels = voxels
        if ignored_labels:
            self.regions = list(set(self.regions) - set(ignored_labels))
            idx = np.in1d(self.labels.ravel(),
                          self.regions).reshape(self.labels.shape)
            reg_x, reg_y, reg_z = np.where(idx)
            self.voxels = np.array(list(zip(reg_x, reg_y, reg_z)))

    def pick_voxels(self, n_voxels):
        raise NotImplementedError

    def pick_all(self):
        raise NotImplementedError


class PickVoxelRandom(PickVoxel):
    def __init__(self, labels, voxels, ignored_labels=None):
        super(PickVoxelRandom, self).__init__(labels,
                                              voxels,
                                              ignored_labels)

    def pick_voxels(self, n_voxels):
        """Randomly pick up voxels regardless of label."""
        rp = np.random.permutation(range(self.voxels.shape[0]))
        idx_voxels = rp[:n_voxels]
        return self.voxels[idx_voxels]

    def pick_all(self):
        return self.voxels


class PickVoxelBalanced(PickVoxel):
    def __init__(self, labels, voxels, ignored_labels=None):
        super(PickVoxelBalanced, self).__init__(labels,
                                                voxels,
                                                ignored_labels)

    def pick_voxels(self, n_voxels, expand_boundary=False):
        """Pick up voxels evently from each region.

        In principle, each region will have n_voxels/n_region voxels.
        If any of the region does not have sufficient voxels,
        the small regions will have duplicated voxels
        to fullfil the number of voxels needed for its region.

        if expand_boundary is set to True,
        Sample background voxels first on the boundary of
        non-background voxels, and random sampling to get
        the rest of required background voxels.

        Note that if the number of voxels is less
        than the number of regions, a random sampling
        regardless of the regions is used.
        """
        n_regions = len(self.regions)
        if n_voxels < n_regions:
            # TODO: get voxels in self.regions
            rp = np.random.permutation(range(self.voxels.shape[0]))
            idx_voxels = rp[:n_voxels]
            return self.voxels[idx_voxels]

        # Distribute the needed voxels to all regions
        n_exp_vx, n_remain = divmod(n_voxels, n_regions)
        # Number of voxels to be extracted
        n_needed_voxels = np.zeros((n_regions,), dtype=int)

        # Sample voxels as expected, leading duplicated voxels
        n_needed_voxels = n_exp_vx * np.ones((n_regions,), dtype=int)

        # randomly choose some non-background regions for remain voxels
        rp = np.random.permutation(len(self.regions))
        rp = rp[rp != 0]
        for reg_id in rp[:n_remain]:
            n_needed_voxels[reg_id] += 1

        boundary_regions = []
        # create boundary of each region
        if expand_boundary:
            nonzero_regions = self.regions.nonzero()[0]
            # TODO: make boundary_width an argument
            boundary_width = 10
            self.labels = create_boundary(self.labels,
                                          nonzero_regions,
                                          boundary_width)
            boundary_regions = list(set(np.unique(self.labels)) -
                                    set(self.regions))

        # Pick up the voxels
        region_voxels = []
        for i, reg_id in enumerate(self.regions):
            n_needed = n_needed_voxels[i]
            reg_indices = np.where(self.labels == reg_id)
            vxs = np.asarray(reg_indices).T
            n_vxs = vxs.shape[0]
            # print("region {} has {}, needs {}".format(i, n_vxs, n_needed))
            # randomly pick as many as it should/could
            rp = np.random.permutation(range(n_vxs))
            sampled_vxs = vxs[rp[:n_needed]]
            region_voxels.extend(sampled_vxs)

            # sample duplicate voxels if region is too small
            if n_needed > n_vxs:
                print("Extract duplicated voxels in region {}".format(i))
                idx_dup_vxs = np.random.randint(n_vxs, size=n_needed - n_vxs)
                dup_vxs = vxs[idx_dup_vxs]
                region_voxels.extend(dup_vxs)

        # extract boundary voxels seperately
        boundary_voxels = []
        for reg_id in boundary_regions:
            reg_indices = np.where(self.labels == reg_id)
            vxs = np.asarray(reg_indices).T
            rp = np.random.permutation(len(vxs))
            sampled_vxs = vxs[rp]
            boundary_voxels.extend(sampled_vxs)

        boundary_voxels = np.asarray(boundary_voxels)

        # replace background voxels with unique boundary voxels
        bg_voxels = region_voxels[:n_needed_voxels[0]]
        rp = np.random.permutation(len(boundary_voxels))
        boundary_voxels = boundary_voxels[rp]

        # from all boundary voxels pick some of them
        if len(boundary_voxels) > len(bg_voxels):
            boundary_voxels = boundary_voxels[:len(bg_voxels)]

        region_voxels[:len(boundary_voxels)] = boundary_voxels
        region_voxels = np.asarray(region_voxels)
        return region_voxels

    def pick_all(self, ratio='auto'):
        """Pick up all voxels.

        Picks up all non-background voxels,
        and the some amount background (if sufficient).

        Arguments:
            ratio: int
                The ratio between non-background and background
                (default = 'auto' = 1/n_regions)
        """
        if ratio == 'auto':
            ratio = 1 / float(len(self.regions))
        nb = self.labels.nonzero()
        nb_voxels = np.array(list(zip(nb[0], nb[1], nb[2])))
        n_nb_voxels = nb_voxels.shape[0]

        bg = np.where(self.labels == 0)
        bg_voxels = np.array(list(zip(bg[0], bg[1], bg[2])))
        rp = np.random.permutation(int(n_nb_voxels * ratio))
        bg_voxels = bg_voxels[rp]
        all_voxels = np.vstack((nb_voxels, bg_voxels))
        return all_voxels

    def pick_voxels_region(self, n_voxels, is_strict=False):
        """Pick voxels per region.

        Arguments:
            n_voxels: int
                The number of voxels from each region.
            is_strict: boolean
                Whether to force the number of sampled voxels
                from each region to be the same.

        By default
        If a region does not contain the required amout of voxels,
        use all the voxels.

        If the argument is_strict is set to True,
        then all the regions will be forced to have the same amount of voxles.
        """
        # Collect the voxels region by region
        reg_all_voxels = []
        num_voxels = np.zeros((len(self.regions),))
        for i, reg_id in enumerate(self.regions):
            reg_indices = np.where(self.labels == reg_id)
            reg_voxels = np.asarray(reg_indices).T
            reg_all_voxels.append(reg_voxels)
            num_voxels[i] = reg_voxels.shape[0]

        # Set up the number of voxels needed for each region
        n_needed_voxels = np.zeros((len(self.regions),)) * n_voxels
        min_num_voxels = num_voxels.min()
        if is_strict and (n_voxels > min_num_voxels):
            n_needed_voxels = np.ones((len(self.regions,))) * min_num_voxels

        # Check how many we can actually extract
        for i in range(len(self.regions)):
            if n_needed_voxels[i] > num_voxels[i]:
                n_needed_voxels[i] = num_voxels[i]

        # Pick up the voxels
        region_voxels = []
        for i in range(len(self.regions)):
            vxs = reg_all_voxels[i]
            rp = np.random.permutation(range(vxs.shape[0]))
            region_voxels.extend(vxs[rp[:n_needed_voxels[i]]])
        region_voxels = np.asarray(region_voxels)
        print("Total voxels sampled = {}".format(region_voxels.shape[0]))
        return region_voxels


class PickVoxelBalancedNonBackground(PickVoxelBalanced):
    """Pick up voxels evenly from each non-background region."""
    def __init__(self, labels, voxels):
        super(PickVoxelBalancedNonBackground, self).__init__(labels,
                                                             voxels,
                                                             ignored_labels=[0])

    def pick_all(self):
        """Pick up all non-backgrond voxels."""
        nb = self.labels.nonzero()
        nb_voxels = np.array(list(zip(nb[0], nb[1], nb[2])))
        return nb_voxels
