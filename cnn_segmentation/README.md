 ## Application
 
This repository implements whole brain segmentation, which segments brain MRI scan into 134 anatomical regions. The methodogy and result have been published in a book chapter: 

[Chapter 10: Characterization of Errors in Deep Learning-Based Brain MRI Segmentation." Deep Learning for Medical Image Analysis](https://www.elsevier.com/books/deep-learning-for-medical-image-analysis/zhou/978-0-12-810408-8).

## Data preprocessing
The raw dataset should be placed in the following structure

```
datasets
└── miccai
  └── train
    ├── mri
    │   ├── 1000_3.nii
    │   ├── 1001_3.nii
    ├── label
    │   ├── 1000_3_glm.nii
    │   ├── 1001_3_glm.nii
 └── test
    ├── mri
    │   ├── 1003_3.nii
    │   ├── 1004_3.nii
    ├── label
    │   ├── 1003_3_glm.nii
    │   ├── 1004_3_glm.nii
```
Then go to ```./preprocessing```

     # convert raw segmentation into 135 classes
     python export_miccai_labels.py


## Code structure and overview

The experiment has been modularized into several modules.

- train_triplanar.py: specify how to sample training voxels and patch features
- network.py: constuct the models using Keras
- cnn_utils: common utilities used when segmenting unseen images
- segmentation_triplanar.py: a wrapper that spawns multiple instances to run prediction workers
- prediction_worker_triplanar.py: the worker function that applies trained network to segment unseen images
- access_segmentation:py compare and evaluate the segmentation result with the ground truth

You may run them separately or use a scipt to combine them and avoid copy-paste errors.

## Reproduce the experiments
Please refer to ```main_pipeline_triplanar.py``` to see how to segment images.

    $ THEANO_FLAGS="device=0 " python main_pipeline_triplanar.py

With the example pipeline, you will use device 0 to train a model,
and segment the images using 3 gpus in parallel.

You should achieve dice score ~0.72 using the example setting in 135 class task.
After that, you can run second run segmentation using the results gained from above

     $ THEANO_FLAGS="device=0 " python main_pipiline_triplanar_cent.py

You should achieve performance ~0.735 ysing the example setting in class task.

Note that the performance is the mean dice of 134 non-background regions.
