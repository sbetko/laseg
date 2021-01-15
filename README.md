## Augmentation Procedure
The augmentation procedure follows a respace-crop-resize operator chain.

- **`Respacing`.** Resample images and masks to common (1 mm x 1 mm x 1mm) spacing. Most images are by default at around a half-millimeter spacing, so this has an overall downsampling effect on the voxel space.
- **`Cropping`.** Crop images/masks to smallest image/mask pair dimensions, preferring to crop regions that do not contain the left atria.
- **`Resizing`.** Resample images and masks to common (64 x 64 x 64) size. This changes the spacing of the images, but to one shared by all images.

The end result is a set of images and masks with equal size and spacing and which contain an unobstructed view of the left atria, possibly at the expense of right-oriented structures for larger hearts. Currently, there are only three images for which this procedure can not preserve the left atria: `586`, `593`, and `679`. These are left out of the analysis.

### Implementation overview

This pipeline is designed to carry out the above steps with minimal runtime, storage, and memory requirements. The implementation details are orchestrated from the `etl.augmentation_pipeline()` driver function in the following sequence:
- Assume a read-only directory containing hi-res image-mask pairs (`img_path` and `msk_path`, respectively), and empty read-write directories for output image-mask pairs (`img_path_out` and `msk_path_out`).
- **(with `n_jobs`)** Read a hi-res file, respace it, and write to file, returning the resultant size.
- **(main `thread`)** Determine the minimum sizing out of all returned sizes from the previous step
- **(with `n_jobs`)** Determine optimal cropping ratios for each image, based on the location of the left atria. See [Dynamic Cropping](#dynamic_cropping) below for an explanation on how such ratios are determined.
- **(with `n_jobs`)** Read and remove a *respaced* file, crop it, and write to file.
- **(with `n_jobs`)** Read and remove a *cropped* file, resize it, and write to file.


### Dynamic Cropping

Given a fixed number of voxels to crop off of each dimension in a 3d image (the **cropping quota**), a naive cropping procedure will drop an equal number of voxels from both the starts and ends of each axis. Hence we can assign a real-valued, scalar **cropping ratio** of `0.5` to each dimension, to represent the proportion of the cropping quota to drop from the start. We call this a *fixed ratio* cropping procedure, where a single ratio for each dimension is determined once and applied to all images.

For fixed ratio cropping to work in practice (i.e., to not risk cropping off regions of interest), we require an unusual condition: all of the bounding boxes of the *imaged structures* (in this case the left atria in each image) must be equidistant to the image boundary along each axis. In other words, there must be a uniform centering of all left atria within the image. Since there is natural centering variation between images, we instead seek to find appropriate cropping ratios for each image individually. We call this *dynamic ratio* cropping, where a heuristic is applied on each image to determine the best apportionment of the cropping quota between the starts and ends of each axis.

![Fixed Ratio Cropping](https://raw.githubusercontent.com/pkla/laseg/master/diagrams/fixed_ratio_cropping.png) ![Dynamic Ratio Cropping](https://raw.githubusercontent.com/pkla/laseg/master/diagrams/dynamic_ratio_cropping.png)