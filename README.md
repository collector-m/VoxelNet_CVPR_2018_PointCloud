# Introduction
This is an unofficial inplementation of [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/abs/1711.06396) in PyTorch.
This project is based on the work (in TensorFlow) [here](https://github.com/qianguih/voxelnet).
Thanks to [@qianguih](https://github.com/qianguih).


# Dependencies
- `python3.5+`
- `Pytorch` (tested on 0.4.1)
- `TensorBoardX` (tested on 1.4)
- `OpenCV`
- `Pillow` (for add_image in TensorBoardX)
- `Boost` (for compiling evaluation code)


# Installation
1. Clone this repository.

2. Compile the Cython module:
```bash
$ python utils/setup.py build_ext --inplace
```

3. Compile the evaluation code:
```bash
$ cd eval/KITTI
$ g++ -I path/to/boost/include -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp
```

4. grant the execution permission to evaluation script:
```bash
$ cd eval/KITTI
$ chmod +x launch_test.sh
```


# Data Preparation
1. Download the 3D KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
Description of annotation can be found [here](https://github.com/yanii/kitti-pcl/blob/master/KITTI_README.TXT). Data to download includes:
    * Velodyne point clouds (29 GB): unzip it and put `'training'` and `'testing'` sub-folders into `'data/KITTI/point_cloud'`
    * Training labels of object data set (5 MB) for input labels of VoxelNet: unzip it and put `'training'` sub-folder into `'data/KITTI/label'`
    * Camera calibration matrices of object data set (16 MB) for visualization of predictions: unzip it and put `'training'` and `'testing'` sub-folders into `'data/KITTI/calib'`
    * Left color images of object data set (12 GB) for visualization of predictions: unzip it and put `'training'` and `'testing'` sub-folders into `'data/KITTI/image'`

2. Crop point cloud data for training and validation. Point clouds outside the image coordinates are removed. Modify data path in `preproc/crop.py` and run it to generate cropped data. Note that cropped point cloud data will overwrite raw point cloud data.

3. Download the train/val split protocal [here](https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz) and untar it into `'data/KITTI'`. Modify data path in `'preproc\split.py'` and run it to generate train/val folder which has the following structure:
```plain
└── data
       ├── KITTI
       └── MD_KITTI
                  ├── training   <-- training data
                  |          ├── image_2
                  |          ├── label_2
                  |          └── velodyne
                  └── validation  <--- evaluation data
                               ├── image_2
                               ├── label_2
                               └── velodyne
```

4. Update the dataset directory in `config.py` and `eval/KITTI/launch_test.sh`


# Train
1. Specify the GPUs to use in `config.py`. Currently, the code only supports single GPU.
2. run `train.sh` with desired hyper-parameters to start training:
```bash
$ bash train.sh
```
Note that more hyper-parameters can be specified in this bash file or in the python file directly.
Another group of hyper-parameters can be found in the [repo](https://github.com/qianguih/voxelnet).

During training, training statistics are recorded in `log/default`, which can be monitored by TensorboardX using command `tensorboard --logdir=path/to/log/default`. Models are saved in `saved/default`.

Intermediate validation results will be dumped into the folder `preds/XXX/data` with `XXX` as the epoch number. Metrics will be calculated (by calling `eval/KITTI/launch_test.sh`) and saved in  `predictions/XXX/log`.

If the `--vis` flag is set to be `True`, visualizations of intermediate results will be dumped in the folder `preds/XXX/vis`.

3. When the training is done, run `parse.sh` to generate the learning curve.
```bash
$ bash parse.sh
```

4. There is a pre-trained model for car in `save_model/pre_trained_car`.


# Evaluate
1. Run `test.sh` to produce final predictions on the validation set after training is done. Change `--tag` flag to `pre_trained_car` will test for the pre-trained model (to be done).
```bash
$ bash test.sh
```
Note taht results will be dumped into `preds/data`. Set the `--vis` flag to True to dump visualizations into `preds/vis`.

2. Run the following command to measure the quantitative performance of predictions:
```bash
$ ./eval/KITTI/evaluate_object_3d_offline ./data/MD_KITTI/validation/label_2 ./preds
```


# TODO
- [ ] Support multiple GPUs
- [ ] reproduce results for `Car`, `Pedestrian` and `Cyclist`


