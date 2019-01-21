# data_object_image_2.zip
Images from the second color camera.

# data_object_velodyne.zip
Velodyne scanned points. Each file (.bin) corresponds to an image and contains around 12K 3D points.
Each point is stored in the format (x, y, z, r), where r is the reflectance value.

# data_object_label_2.zip
Label format can be found [here](https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md#label-format).
The last value is only used for online submission. Original description can be downloaded [here](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip).



# data_object_calib.zip
Pi: projection matrix after rectification, size of 3x4; i is camera index

R0_rect: rectifying rotation matrix of the reference camera (camera 0)

Tr_velo_to_cam: rotation and translation matrix from the Velodyne coordinate to the camera coordinate

To transform a 3D point x in Velodyne coordinates to a point y in i-th camera image using: **y = Pi * R0_rect * Tr_velo_to_cam * x**