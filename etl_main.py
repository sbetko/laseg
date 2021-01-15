import os

# LOCAL DIRECTORY TREE STRUCTURE

#########################
# \_ la-seg             #  <- GCS Bucket
#    |_  THIS FILE      #
#     \_ hi-res         #  <- Fetched from GCS
#         |_ masks      #
#           \ ...       #
#         \_ images     #
#           \ ...       #
#     \_ lo-res         #  -> Uploaded to GCS
#         |_ masks64    #
#           \ ...       #
#         \_ images64   #
#           \ ...       #
#########################

### DEFINE PATHS
data_path = './la-seg/'                                     # GLOBALVAR

hi_res_path = os.path.join(data_path, 'hi-res')             # GLOBALVAR
hi_res_images_path = os.path.join(hi_res_path, 'images')    # GLOBALVAR
hi_res_masks_path = os.path.join(hi_res_path, 'masks')      # GLOBALVAR

lo_res_path = os.path.join(data_path, 'lo-res')             # GLOBALVAR
lo_res_images_path = os.path.join(lo_res_path, 'images64')  # GLOBALVAR
lo_res_masks_path = os.path.join(lo_res_path, 'masks64')    # GLOBALVAR

def safe_mkdir(path):
    if isinstance(path, list):
        [safe_mkdir(p) for p in path]
    elif not os.path.exists(path):
        os.mkdir(path)

safe_mkdir([data_path,
            hi_res_path, hi_res_images_path, hi_res_masks_path,
            lo_res_path, lo_res_images_path, lo_res_masks_path])

# if __name__ == '__main__':
#     from etl import get_images

#     num_imgs = 45

#     sample_images_hi_res = [] #get_images(hi_res_images_path, n=num_imgs)
#     sample_masks_hi_res = get_images(hi_res_masks_path, n=num_imgs)

#     from etl import resampling_pipeline
#     aug_masks = resampling_pipeline(sample_images_hi_res,
#                                     sample_masks_hi_res,
#                                     masks_only=True,
#                                     n_jobs=1)


if __name__ == '__main__':
    import etl

    from time import time

    start = time()

    etl.augmentation_pipeline(hi_res_images_path, hi_res_masks_path,
                              lo_res_images_path, lo_res_masks_path,
                              n_jobs=16)

    print(f'Finished in {time() - start:.1f} s')