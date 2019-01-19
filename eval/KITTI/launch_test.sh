#!/usr/bin/env bash
# Crooped gt dir
GT_DIR=./data/MD_KITTI/validation/label_2
# pred dir
PRED_DIR=$1

# Output log
OUTPUT=$2
# Start test
nohup `pwd`/eval/KITTI/evaluate_object_3d_offline $GT_DIR $PRED_DIR > $OUTPUT 2>&1 &
