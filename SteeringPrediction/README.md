# Guide to Steering Prediction on the DDD20 Dataset

## Dataset Preparation

1. While the DDD20 dataset is very big, we only need 30 of the raw recordings. The names of these 30 recordings are given [here](https://github.com/SensorsINI/ddd20-utils/blob/master/export-data-ral.sh). Please download these 30 raw recordings from [this link](https://sites.google.com/view/davis-driving-dataset-2020/home).

2. Please use [ddd20-utils](https://github.com/SensorsINI/ddd20-utils) to convert the raw recordings into standard data types. We need to edit the absolute paths in [export-data-ral.sh](https://github.com/SensorsINI/ddd20-utils/blob/master/export-data-ral.sh) and run `bash export-data-ral.sh`. After this script completes, please move all the `*_export.hdf5` files to `data/hdf5/`. The directory structure should look like this:
```
SteeringPrediction
  |-- data
  |     |-- hdf5
  |     |     |-- rec1498946027_export.hdf5
  |     |     |-- rec1498949617_export.hdf5
  |     |     |-- ...
  |     |     |-- <a total of 30 .hdf5 files>
  |     |-- generate_meta.py
  |-- baseline
  |     |-- <implementation of the baseline model>
  |-- ours
  |     |-- <implementation of our model>
```

3. In `./data`, please run `python generate_meta.py`. This creates `./data/meta.pkl` with information of training/testing splits.

## Running the Baseline
Unfortunately, there is not an open-source implementation for the prediction model described in Section III of the DDD20 paper. `./baseline` contains an implementation with our best understanding. To use the baseline model, please first create a symbolic link to the data directory:
```
cd baseline
ln -s ../data .
```

To train the model, please run:
```
python src/train.py --seed <your choice of random seed>
```

Alternatively, you can download our pre-trained weights with five different ramdom seeds here: [Google Drive](https://drive.google.com/file/d/1WIXWEg0wIol6tlZi6Ni8w5fOKp0HgNY_/view?usp=sharing).

To test the model, please run:
```
python src/train --load_dir saved_weights/<name>/checkpoints/0.001/199
```

## Running Our Model
To use our model, please first create a symbolic link to the data directory:
```
cd ours
ln -s ../data .
```

To train the model, please run:
```
python src/train.py --seed <your choice of random seed> --norm <'constant' or 'none'>
```

For example, to train with random seed `0` and integral normalization enabled, please do:
```
python src/train.py --seed 0 --norm constant
```

Alternatively, you can download our pre-trained weights with five different ramdom seeds here: [Google Drive](https://drive.google.com/file/d/1WIXWEg0wIol6tlZi6Ni8w5fOKp0HgNY_/view?usp=sharing).

To test the model, please run:
```
python src/train --load_dir saved_weights/<name>/checkpoints/0.001/199 --norm <'constant' or 'none'>
```

For example:
```
python src/train --load_dir saved_weights/constant_seed0/checkpoints/0.001/199 --norm constant
```
