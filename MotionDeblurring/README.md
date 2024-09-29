# Guide to Motion Deblurring on the HQF Dataset

## Dataset Preparation
Please follow [the instructions from DeblurSR](https://github.com/chensong1995/DeblurSR/blob/main/HQF_Dataset.md) to create 14 .hdf5 files. Move these .hdf5 files to `./data/HQF/`. The final directory structure should look like this:
```
MotionDeblurring
  |-- data
  |     |-- HQF
  |     |     |-- bag
  |     |     |     |-- bike_bay_hdr.bag
  |     |     |     |-- boxes.bag
  |     |     |     |-- <many other .bag files>
  |     |     |     |-- still_life.bag
  |-- <other files>
```

## Running the Model

To train the model with PPLN and constant normalization, please run:
```
python src/train.py --constant_norm 1 --sconv 1 --save_dir saved_weights/constant
```

To train the model with PPLN and without constant normalization, please run:
```
python src/train.py --constant_norm 0 --sconv 1 --save_dir saved_weights/none
```

To train a large U-Net without PPLN layers, please run:
```
python src/train.py --constant_norm 0 --sconv 0 --save_dir saved_weights/unet
```

Alternatively, you can download our pre-trained weights here: [Google Drive](https://drive.google.com/file/d/1WIXWEg0wIol6tlZi6Ni8w5fOKp0HgNY_/view?usp=sharing).

To test the model, please run:
```
python src/train.py --constant_norm 1 --sconv 1 --load_dir saved_weights/constant/checkpoints/49
```

Or:
```
python src/train.py --constant_norm 0 --sconv 1 --load_dir saved_weights/none/checkpoints/49
```

Or:
```
python src/train.py --constant_norm 0 --sconv 0 --load_dir saved_weights/unet/checkpoints/49
```
