import nibabel as nib
import glob
import time
from functools import partial
from utils.utils import (
    change_parent_dir,
    distribute_samples,
    parmap_star,
    run_theano_script,

)


# Inner functions
def predict_with_worker(gpu, img_pathes, mask_pathes,
                        prediction_worker,
                        model_path, stats_path,
                        n_classes, patch_size, scales,
                        batch_size, gpu_batch_size,
                        out_dir):
    """Send argugment to an prediction worker script to segment images.

    Arguments:
        gpu: string
            The gpu id to run on (e.g. 'gpu0', 'cuda0')
        img_pathes: list of strings
            The list of image path
        mask_pathes: list of strings
            The list of corresponding mask path
        prediction_worker: string
            The path of prediction worker script, which takes all the arugments
            in input, with gpu id as preload config
        model_path: string
            The trained network model
        stats_path: string
            The statistcis of model, needed for segmentating
        n_classes: int
           The number of classes including background
        patch_size: int
            The input patch size
        scales: list of int
            The scales used
        batch_size: int
            The batch size used to extract patch feature by batches
        gpu_batch_size: int
           The batch size fed into gpu
        out_dir: string
            The output directory for segmentation results.

    Remember to put the private arguments first
    if you wish to use itertools.partial.
    """
    # concatenate the arguments
    if not img_pathes:
        return
    args = []
    args.append(' '.join(img_pathes))
    args.append(' '.join(mask_pathes))
    args.append(model_path)
    args.append(stats_path)
    args.append(n_classes)
    args.append(patch_size)
    args.append(' '.join(map(str, scales)))
    args.append(batch_size)
    args.append(gpu_batch_size)
    args.append(out_dir)

    # run worker with specified gpu
    run_theano_script(prediction_worker, gpu, args)


# APIs
def segment_triplanar(img_pathes, mask_pathes,
                      prediction_worker,
                      model_path, stats_path,
                      n_classes, patch_size, scales,
                      batch_size, gpus, gpu_batch_size,
                      out_dir):
    """Segmenta images with multiples devices.

    Arguments:
        img_pathes: list of strings
            The list of image path
        mask_pathes: list of string
            The list of corresponding mask path
        prediction_worker: string
            The path of prediction worker script, which takes all the arugments
            in input, with gpu id as preload config
        model_path: string
            The trained network model
        stats_path: string
            The statistcis of model, needed for segmentating
        n_classes: int
            The number of classes including background
        patch_size: int
            The input patch size
        scales: list of int
            The scales used
        batch_size: int
            The batch size used to extract patch feature by batches
        devices: list of string
            The devices id to run on (e.g. 'gpu0', 'cuda0', 'cpu')
        gpu_batch_size: int
           The batch size fed into gpu
        out_dir: string
            The output directory for segmentation results.

    Split the test images into several subsets.
    Each device handles one subset and spawns one worker to process the images.

    The worker script applies the same parameters
    to several test images and their mask.
    """
    if len(img_pathes) == 0:
        return

    # split (image, mask) pairs into several gpus
    atlas_subsets = distribute_samples(list(zip(img_pathes, mask_pathes)),
                                       len(gpus))
    img_subsets = []
    mask_subsets = []
    for atlas in atlas_subsets:
        img_subset, mask_subset = zip(*atlas)
        img_subsets.append(img_subset)
        mask_subsets.append(mask_subset)

    # set shared arguments
    partial_predict = partial(predict_with_worker,
                              prediction_worker=prediction_worker,
                              model_path=model_path,
                              stats_path=stats_path,
                              n_classes=n_classes,
                              patch_size=patch_size,
                              scales=scales,
                              batch_size=batch_size,
                              gpu_batch_size=gpu_batch_size,
                              out_dir=out_dir)

    parmap_star(partial_predict, zip(gpus, img_subsets, mask_subsets))


if __name__ == '__main__':
    nib.nifti1.Nifti1Header.quaternion_threshold = -1e-6
    test_img_dir = './datasets/miccai/test/mri/'
    mask_dir = './datasets/miccai/test/label_mat/'
    test_img_pathes = glob.glob(test_img_dir+'*.nii')
    test_mask_pathes = [change_parent_dir(mask_dir, f, '_glm.mat')
                        for f in test_img_pathes]
    prediction_worker = './prediction_worker_triplanar.py'
    model_path = './experiments/keras/triplanar/cnn_triplanar.h5'
    stats_path = './experiments/keras/triplanar/cnn_triplanar_stat.h5'
    n_classes = 135
    patch_size = 29
    scales = [1, 3]
    batch_size = 200000
    gpu_batch_size = 40000
    out_dir = './experiments/keras/triplanar/'
    gpus = ['gpu0', 'gpu1', 'gpu2', 'gpu3']

    start_time = time.time()
    segment_triplanar(test_img_pathes, test_mask_pathes,
                      prediction_worker,
                      model_path, stats_path,
                      n_classes, patch_size, scales,
                      batch_size, gpus, gpu_batch_size,
                      out_dir)
    print("Done in {} seconds".format(time.time()-start_time))
