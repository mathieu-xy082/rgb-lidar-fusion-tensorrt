# Data

Do not commit datasets to this repository.

Recommended starting point: KITTI object detection dataset.

Expected local layout, once downloaded manually:

```text
data/kitti/
  training/
    image_2/
    velodyne/
    calib/
    label_2/
  testing/
    image_2/
    velodyne/
    calib/
```

The first milestone only needs a few calibration files, RGB images, and Velodyne point clouds to validate projection.
