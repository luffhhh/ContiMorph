# ContiMorph: An Unsupervised Learning Framework for Cardiac Motion Tracking with Time-continuous  Diffeomorphism
This repository is the official implementation of "ContiMorph: An Unsupervised Learning Framework for Cardiac Motion Tracking with Time-continuous  Diffeomorphism".
![Image1](img/figure3.png)
![Gif1](img/AC_inf.gif)
## Environment
- Please prepare an environment with python=3.9, run the following command
```
pip install -r requirements.txt
```

## Dataset
In our experiments, we used the following datasets:
* 3D cardiac MR images: [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
* 3D cardiac MR images: [M\&Ms dataset](https://www.ub.edu/mnms/)
* 2D ultrasound images: [CAMUS dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html)

    ```
    ├─train
        ├─patient01
        ├ ├─sequence1.nii.gz
        ├ ├─sequence2.nii.gz
        ├ └─...
        ├─patient01
        ├ ├─sequence1.nii.gz
        ├ ├─sequence2.nii.gz
        ├ └─...
        ├─patient01
        ├ ├─sequence1.nii.gz
        ├ ├─sequence2.nii.gz
        ├ └─...
        ├─...
        └─cine_files.txt
    ├─val
        ├─patient01
        ├ ├─sequence1.nii.gz
        ├ ├─sequence2.nii.gz
        ├ └─...
        ├─patient01
        ├ ├─sequence1.nii.gz
        ├ ├─sequence2.nii.gz
        ├ └─...
        ├─patient01
        ├ ├─sequence1.nii.gz
        ├ ├─sequence2.nii.gz
        ├ └─...
        ├─...
        └─cine_files.txt
    ├─test
        ├─patient01
        ├ ├─sequence1.nii.gz
        ├ ├─sequence2.nii.gz
        ├ └─...
        ├─patient01
        ├ ├─sequence1.nii.gz
        ├ ├─sequence2.nii.gz
        ├ └─...
        ├─patient01
        ├ ├─sequence1.nii.gz
        ├ ├─sequence2.nii.gz
        ├ └─...
        ├─...
        └─cine_files.txt
    ```

## Training
* To train the model run the following:
```
python ext/train_the_net.py
```

## Evaluation
* To evaluate the model run the following:
```
python ext/test_the_net.py
```

## Pre-trained model
```
model/pro_new_model.pth
```
## Acknowledgments
Our code implementation borrows heavily from [VoxelMorph](https://github.com/voxelmorph/voxelmorph),[Deeptag](https://github.com/DeepTag/cardiac_tagging_motion_estimation/tree/main) and [SGDIR](https://github.com/mattkia/SGDIR/tree/master).
