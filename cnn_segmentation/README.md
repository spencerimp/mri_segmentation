## Intro

This repository implements whole brain segmentation, which segments brain MRI scan into 134 anatomical regions. The methodogy and result have been published in a book chapter:

[Chapter 10: Characterization of Errors in Deep Learning-Based Brain MRI Segmentation." Deep Learning for Medical Image Analysis](https://www.elsevier.com/books/deep-learning-for-medical-image-analysis/zhou/978-0-12-810408-8).


You can request the dataset [here](http://masiweb.vuse.vanderbilt.edu/workshop2012/index.php/Main_Page)

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
- segmentation_triplanar.py: a wrapper that spawns multiple instances to run prediction workers with Theano backend
- prediction_worker_triplanar.py: the worker function that applies trained network to segment unseen images
- access_segmentation:py compare and evaluate the segmentation result with the ground truth

You may run them separately or use a script to combine them and avoid copy-paste errors.

## Reproduce the experiments
Please refer to ```main_pipeline_triplanar.py``` to see how to segment images.

    $ THEANO_FLAGS="device=gpu0 " python main_pipeline_triplanar.py

With the example pipeline, you will use device 0 to train a model,
and segment the images using 3 gpus in parallel.

You should achieve dice score ~0.72 using the example setting.
After that, you can launch second round segmentation using the results gained from above

     $ THEANO_FLAGS="device=gpu0 " python main_pipeline_triplanar_cent.py

You should achieve mean dice score ~0.735 using the example setting.

Note

- The performance is the mean dice of 134 non-background regions.
- At the time of writing, using TensorFlow backend (v1.0) for training yields bad model, use it with caution for this application.
