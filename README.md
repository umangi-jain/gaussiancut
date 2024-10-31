# GaussianCut

This is the repo to run GaussianCut for selecting objects in a scene represented with 3DGS.

We take the user inputs for segmentation through the UI of [Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything). We operate directly on a pretrained 3DGS model optimized using the official [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting/) repo. Model pipeline:

(a) Select the foreground object and propogate it to obtain multiview masks using SAM-and-Track \
(b) Use an optimized 3DGS model for the scene \
(c) Apply GaussianCut to obtain finer segments of the object of interest 

### Repository Setup:

Create an environment and install the following dependencies:

- Clone this repository
```
git clone https://github.com/umangi-jain/gaussiancut.git
# ensure the pytorch and cuda versions are compatible
conda env create -f environment.yml
```
- Prepare the dataset (similar to 3DGS) directory. Get an optimized 3DGS model (please see the [3DGS](https://github.com/graphdeco-inria/gaussian-splatting/) repo for further details).
- Install dependencies for the video segmentation model used to get multiview masks. We use [Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything). Either use a camera trajectory to generate a video or SAM-Track can also handle unordered images directly. Select the object of interest and save the masks. Please note the number of masks can be decided arbitary (a single mask is also okay).
- Apply graph cut
```
bash scripts/path_to_a_scene_config.sh
```

Results for the coarse and fine splatting would be saved under the dataset directory.


### Directory Structure
The directory structure for the dataset is as follows:
```
scene
├── optimized_3dgs_models
│   ├── model
│       ├── coarse_results
│       └── fine_results
│       └── graphcut_weights
│       └── point_cloud
└── input
    ├── IMG_001.png
    ├── IMG_002.png
    ├── ...
    ├── IMG_013.png
└── multiview_masks
    ├── IMG_001.png
    ├── ...
    └── IMG_010.png
```

### Acknowledgment
Our work builds on top of these amazing repositories:
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything)
- [GaussianEditor](https://github.com/buaacyw/GaussianEditor)