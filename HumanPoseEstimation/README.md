# Guide to Human Pose Estimation on the DHP19 Dataset

## Dataset Preparation

1. Please download the DHP19 dataset from [this link](https://sites.google.com/view/dhp19/download).

2. Please use the MATLAB data processing script in [this GitHub repository](https://github.com/SensorsINI/DHP19) to create .h5 files. Move the folders to `./data` to make the directory structure look like this
```
HumanPoseEstimation
  |-- data
  |     |-- h5_dataset_7500_events
  |     |     |-- 346x260
  |     |     |     |-- <many .h5 files>
  |     |     |-- Fileslog_<timestamp>.log
  |     |-- P_matrices
  |     |     |-- <a few camera calibration files>
  |     |-- merge_h5.py
  |-- baseline
  |     |-- <implementation of the baseline model>
  |-- ours
  |     |-- <implementation of our model>
```

3. In `./data`, please run `python merge_h5.py`. This merges all .h5 files belonging to the same human subject into one single file. This command creates a directory `data/processed` with 17 .hdf5 files.

## Running the Baseline
Unfortunately, there is not an open-source implementation for the prediction model described in Section 4 of the DHP19 paper. `./baseline` contains an implementation with our best understanding. To use the baseline model, please first create a symbolic link to the data directory:
```
cd baseline
ln -s ../data .
```

To train the model, please run:
```
python src/train.py
```

Alternatively, you can download our pre-trained weights here: [Google Drive](https://drive.google.com/file/d/1WIXWEg0wIol6tlZi6Ni8w5fOKp0HgNY_/view?usp=sharing).

To test the model, please run:
```
python src/test3d.py --pred_name saved_weights/release/keypoints.npy
```

## Running Our Model
To use our model, please first create a symbolic link to the data directory:
```
cd ours
ln -s ../data .
```

To train the model, please run:
```
python src/train.py --norm <'constant' or 'none'>
```

Alternatively, you can download our pre-trained weights here: [Google Drive](https://drive.google.com/file/d/1WIXWEg0wIol6tlZi6Ni8w5fOKp0HgNY_/view?usp=sharing).

To test the model, please run:
```
python src/test3d.py --pred_name saved_weights/<'constant' or 'none'>/keypoints.npy
```
