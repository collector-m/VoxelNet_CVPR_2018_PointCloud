import os
from shutil import copyfile

import pdb


def rearrange(original_folders, target_folders, split_file):
    for _, v in original_folders.items():
        if not os.path.isdir(v):
            raise Exception('No such folder: %s' % v)

    for _, v in target_folders.items():
        if not os.path.isdir(v):
            os.makedirs(v)

    if not os.path.exists(split_file):
        raise Exception('No such split file: %s' % split_file)
    else:
        with open(split_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        for file in content:
            src_img = os.path.join(original_folders['IMG_ROOT'], file + '.png')
            src_pc = os.path.join(original_folders['PC_ROOT'], file + '.bin')
            src_label = os.path.join(original_folders['LABEL_ROOT'], file + '.txt')

            if (not os.path.exists(src_img)) or (not os.path.exists(src_pc)) or (not os.path.exists(src_label)):
                 print('No such file: %s' % file)
            else:
                dst_img = os.path.join(target_folders['IMG_ROOT'], file + '.png')
                dst_pc = os.path.join(target_folders['PC_ROOT'], file + '.bin')
                dst_label = os.path.join(target_folders['LABEL_ROOT'], file + '.txt')

                copyfile(src_img, dst_img)
                copyfile(src_pc, dst_pc)
                copyfile(src_label, dst_label)


if __name__ == '__main__':
    # Original folder
    original_folders = dict()
    original_folders['IMG_ROOT'] = './data/KITTI/image/training/image_2/'
    original_folders['PC_ROOT'] = './data/KITTI/point_cloud/training/velodyne/'
    original_folders['LABEL_ROOT'] = './data/KITTI/label/training/label_2/'

    # Modified folder
    train_folders = dict()
    train_folders['IMG_ROOT'] = './data/MD_KITTI/training/image_2/'
    train_folders['PC_ROOT'] = './data/MD_KITTI/training/velodyne/'
    train_folders['LABEL_ROOT'] = './data/MD_KITTI/training/label_2/'

    val_folders = dict()
    val_folders['IMG_ROOT'] = './data/MD_KITTI/validation/image_2/'
    val_folders['PC_ROOT'] = './data/MD_KITTI/validation/velodyne/'
    val_folders['LABEL_ROOT'] = './data/MD_KITTI/validation/label_2/'

    # Split file
    train_split_file = './data/KITTI/imagesets/ImageSets/train.txt'
    val_split_file = './data/KITTI/imagesets/ImageSets/val.txt'

    rearrange(original_folders, train_folders, train_split_file)
    rearrange(original_folders, val_folders, val_split_file)