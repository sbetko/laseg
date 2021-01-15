import SimpleITK as sitk
import pandas as pd
import numpy as np
import os

from zipfile import ZipFile
import gzip
import glob

from tqdm import tqdm

import multiprocessing as mp

def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    itk_image.SetOrigin([0, 0, 0])
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    return resample.Execute(itk_image)

def resample_image_standardize(itk_image, out_size=(64, 64, 64), is_label=False):
    itk_image.SetOrigin([0, 0, 0])
    
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_spacing = [(original_size[0] * (original_spacing[0] / out_size[0])),
                   (original_size[1] * (original_spacing[1] / out_size[1])),
                   (original_size[2] * (original_spacing[2] / out_size[2]))]

    reference_image = sitk.Image(out_size, 1)
    reference_image.SetDirection(itk_image.GetDirection())
    reference_image.SetSpacing(out_spacing)
    reference_image.SetPixel 
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline

    return sitk.Resample(itk_image, reference_image, sitk.Transform(), interpolator)

def reslice_image(itk_image, itk_ref, is_label=False):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(itk_ref)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    return resample.Execute(itk_image)

def normalize(image):
    MIN_BOUND = np.min(image)
    MAX_BOUND = np.max(image)
    image = (image - MIN_BOUND).astype(float)/(MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def safe_mkdir(path):
    if isinstance(path, list):
        [safe_mkdir(p) for p in path]
    elif not os.path.exists(path):
        os.mkdir(path)

def validate_bucket_download(hi_res_images_path, hi_res_masks_path):
    ### LOOK FOR MASKS AND IMAGE FILES
    masks_list = os.listdir(hi_res_masks_path)
    images_list = os.listdir(hi_res_images_path)

    ### VERIFY DATA INTEGRITY
    num_masks = len(masks_list)
    num_images = len(images_list) 

    ### 1. VERIFY MATCHING NUMBER OF MASKS AND IMAGES
    if num_masks != num_images:
        raise FileNotFoundError(f"Unequal number of masks and images.")

    masks_list.sort()
    images_list.sort()

    image_mask_pairs = list(zip(images_list, masks_list))

    ### 2. VERIFY 1-1 CORRESPONDENCE BETWEEN MASKS AND IMAGES
    for pair in image_mask_pairs:
        name_1 = ''.join(pair[1].split('.')[0].split('-')[1:])
        name_2 = ''.join(pair[0].split('.')[0].split('-')[1:])
        if (name_1 != name_2):
            msg = f'Incomplete correspondence between masks and images: '
            msg += f'found non-matching (mask, image) pair {name_1, name_2} from files {pair}.'
            raise FileNotFoundError(msg)
    
    return True

# For information about the image types see the SimpleITK docs:
# https://simpleitk.readthedocs.io/en/master/IO.html

def resample_directory_helper(args):
    """Helper function for resample_directory(). Resamples a single medical image and writes to the output path.

    Arguments:
        args {list} -- Packed args tuple containing the following:
        file_name {str} -- Name of file or folder to resample
        input_file_path {str} -- Path to file or folder to resample
        output_file_path {str} -- Path to output file to write resampled image to
        out_size {tuple} -- Shape to resample image to
        is_label {bool} -- Determines whether to use the NearestNeighbor resampling function (True) or a B-spline (False)
        image {str} -- The type of SimpleITK ImageIO reader to use 
        output_type {str} -- If the input files or folders do not have an extension, this will determine the output extension type
    """
    file_name, input_file_path, output_file_path, out_size, is_label, image, output_type = args

    # If the input image does not have a file extension specified,
    # then append the specified file extension to the output filepath
    # to be interpreted by sitk.WriteImage() later.
    if '.' not in file_name:
        output_file_path = output_file_path + '.' + output_type

    if image == 'GDCMImageIO':
        reader = sitk.ImageSeriesReader()

        dicom_names = reader.GetGDCMSeriesFileNames(input_file_path)
        reader.SetFileNames(dicom_names)

        original = reader.Execute()

    elif image == 'NiftiImageIO':
        original = sitk.ReadImage(input_file_path, imageIO=image)

    resampler = resample_image_standardize(original, out_size=out_size, is_label=is_label)

    sitk.WriteImage(resampler, output_file_path)
    

def resample_directory(input_directory, output_directory, out_size=(64,64,64), is_label=False, image=None, output_type='nii', n_jobs=None):
    """Resamples a given input directory of medical images and outputs to the specified output directory

    Arguments:
        input_directory {list} -- Directory to read images from
        output_directory {list} -- Directory to write resampled images to

    Keyword Arguments:
        out_size {tuple} -- The shape of the resampled output (default: {(64,64,64)})
        is_label {bool} -- Determines whether to use the NearestNeighbor resampling function (True) or a B-spline (False) (default: {False})
        image {str} -- The type of SimpleITK ImageIO reader to use (default: {None})
        output_type {str} -- If the input files or folders do not have an extension, this will determine the output extension type (default: {'nii'})
        n_jobs {int} -- Number of CPU cores to use. `None` falls back to sequential operation and -1 will use all available cores (default: {None})
    """

    if n_jobs == 0 or n_jobs < -1:
        raise ValueError(f'Illegal n_jobs argument value {n_jobs}. Legal values are None, -1, and n >= 1.')

    args = []
    
    for file_name in os.listdir(input_directory):
        
        input_file_path = os.path.join(input_directory, file_name)
        output_file_path = os.path.join(output_directory, file_name)

        args.append((file_name, input_file_path, output_file_path, out_size, is_label, image, output_type))

    if n_jobs is not None or n_jobs == 1:
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        elif n_jobs > mp.cpu_count():
            n_jobs = mp.cpu_count()

        with mp.Pool(processes=n_jobs) as p:
            list(tqdm(p.imap(resample_directory_helper, args), total=len(args)))
    else:
        for arg in tqdm(args):
            resample_directory_helper(arg)


def gunzip(source_filepath, dest_filepath, block_size=65536):
    with gzip.open(source_filepath, 'rb') as s_file, \
            open(dest_filepath, 'wb') as d_file:
        while True:
            block = s_file.read(block_size)
            if not block:
                break
            else:
                d_file.write(block)


def unzip_glob(path, extension, remove_originals=False):
    
    extension = ''.join(extension.split('.'))
    files = glob.glob(os.path.join(path, f'*.{extension}'))

    for i, _ in enumerate(files):
        destination = files[i].split('.' + extension)[0]
        if extension == 'zip':
            with ZipFile(files[i], 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(path=destination)

        elif extension == 'gz':
            gunzip(files[i], destination)

        if remove_originals:
            os.remove(files[i])

def get_images(directory, n = 5):
    """Returns sitk.Image objects from the directory, sorted lexicographically
    by filename.

    Arguments:
        directory {[type]} -- Path to directory containing images

    Keyword Arguments:
        n {int} -- Read the first 'n' images from directory (default: {5})

    Returns:
        list -- List of sitk.Image objects
    """
    
    paths = [os.path.join(directory, name) for name in sorted(os.listdir(directory))]
    n = min(len(paths), n) if n != -1 else len(paths)
    paths = paths[:n]
    
    imgs = [sitk.ReadImage(path) for path in paths]
    return imgs

# TODO: Rewrite to take a list of (w, h, l) optionally instead of images
def find_shape_extrema(imgs):
    # (w)idth, (h)eight, (l)ength corresponds to maximum number of voxels along the
    # x, y, z axis for a given picture

    # Starting values
    w_max = 0
    h_max = 0
    l_max = 0

    w_min = 1000
    h_min = 1000
    l_min = 1000

    for img in imgs:
        w, h, l = img.GetSize()

        w_max = w if w > w_max else w_max
        h_max = h if h > h_max else h_max
        l_max = l if l > l_max else l_max

        w_min = w if w < w_min else w_min
        h_min = h if h < h_min else h_min
        l_min = l if l < l_min else l_min

    return (w_min, h_min, l_min), (w_max, h_max, l_max)

# def crop_images_to_common_dimensions(imgs):

#     (w_min, h_min, l_min), (w_max, h_max, l_max) = find_shape_extrema(imgs)

#     new_imgs = []

#     width_ratio = 0.5
#     height_ratio = 0.5
#     length_ratio = 0.5

#     for img in tqdm(imgs):
#         w_img = img.GetWidth()
#         h_img = img.GetHeight()
#         l_img = img.GetDepth()

#         # These are the number of voxels to crop off each dimension,
#         # in the proportions dictated by [width/height/length]_ratio
#         w_diff_im = w_img - w_min
#         h_diff_im = h_img - h_min
#         l_diff_im = l_img - l_min

#         i_start = int(w_diff_im // (1 / width_ratio))
#         i_end = int(w_img - w_diff_im // (1 / (1 - width_ratio)))

#         j_start = int(h_diff_im // (1 / height_ratio))
#         j_end = int(h_img - h_diff_im // (1 / (1 - height_ratio)))

#         k_start = int(l_diff_im // (1 / length_ratio))
#         k_end = int(l_img - l_diff_im // (1 / (1 - length_ratio)))

#         print(f'{i_start:3}:{i_end:3}, {j_start:3}:{j_end:3}, {k_start:3}:{k_end:3}; {w_img:3} - {w_diff_im:3}, {h_img:3} -  {h_diff_im:3}, {l_img:3} - {l_diff_im:3}')
#         new_imgs.append(img[i_start:i_end, j_start:j_end, k_start:k_end])

#     return new_imgs

def threshold_based_crop(image, inside_value=0, outside_value=255):
    """Use Otsu's threshold estimator to separate background and foreground.
    Then crop the image using the foreground's axis aligned bounding box.
    Arguments:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Returns:
        Cropped image based on foreground's axis aligned bounding box.                                 
    """
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute( sitk.OtsuThreshold(image, inside_value, outside_value) )
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    return sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])


def crop_images_to_common_dimensions(imgs, fixed_bounding_ratios = (0.5, 0.5, 0.5), dynamic_bounding_ratios = [],
                                    find_dynamic_bounding_ratios=False):

    (w_min, h_min, l_min), (w_max, h_max, l_max) = find_shape_extrema(imgs)

    new_imgs = []

    for i, img in tqdm(enumerate(imgs)):
        print(i, end=': ')
        if dynamic_bounding_ratios == []:
            width_ratio, height_ratio, length_ratio = fixed_bounding_ratios
        else:
            width_ratio, height_ratio, length_ratio = dynamic_bounding_ratios[i]

        w_img = img.GetWidth()
        h_img = img.GetHeight()
        l_img = img.GetDepth()

        # These are the number of voxels to crop off each dimension,
        # in the proportions dictated by [width/height/length]_ratio
        w_diff_im = w_img - w_min
        h_diff_im = h_img - h_min
        l_diff_im = l_img - l_min

        if find_dynamic_bounding_ratios:
            label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
            label_shape_filter.Execute(sitk.OtsuThreshold(img, 0, 1))
            bounding_box = label_shape_filter.GetBoundingBox(1)

            i_start_max, j_start_max, k_start_max = bounding_box[:3]
            i_span_min, j_span_min, k_span_min = bounding_box[3:]

            i_end_min = i_start_max + i_span_min
            j_end_min = j_start_max + j_span_min
            k_end_min = k_start_max + k_span_min

            if (l_diff_im > l_img - k_span_min
                or h_diff_im > h_img - j_span_min
                or w_diff_im > w_img - i_span_min):
                print(f'Bounding box on mask {i} does not meet minimum size requirement for cropping; cropping will trespass minimum bounding box.')
            
            i_end_min_from_end = w_img - i_end_min
            j_end_min_from_end = h_img - j_end_min
            k_end_min_from_end = l_img - k_end_min

            if i_start_max > i_end_min_from_end:
                a_i = i_end_min_from_end
                b_i = i_start_max
                width_ratio = max((a_i / b_i) / 2, 1 - (a_i / b_i) / 2)
                print(0, end=', ')
            else:
                b_i = i_end_min_from_end
                a_i = i_start_max
                width_ratio = min((a_i / b_i) / 2, 1 - (a_i / b_i) / 2)
                print(1, end=', ')
            
            if j_start_max > j_end_min_from_end:
                a_j = j_end_min_from_end
                b_j = j_start_max
                height_ratio = max((a_j / b_j) / 2, 1 - (a_j / b_j) / 2)
                print(0, end=', ')
            else:
                b_j = j_end_min_from_end
                a_j = j_start_max
                height_ratio = min((a_j / b_j) / 2, 1 - (a_j / b_j) / 2)
                print(1, end=', ')
            
            if k_start_max > k_end_min_from_end:
                a_k = k_end_min_from_end
                b_k = k_start_max
                length_ratio = max((a_k / b_k) / 2, 1 - (a_k / b_k) / 2)
                print(0)
            else:
                b_k = k_end_min_from_end
                a_k = k_start_max
                length_ratio = min((a_k / b_k) / 2, 1 - (a_k / b_k) / 2)
                print(1)

        i_start = int(w_diff_im * width_ratio)
        i_end = int(w_img - w_diff_im * (1 - width_ratio))

        j_start = int(h_diff_im * height_ratio)
        j_end = int(h_img - h_diff_im * (1 - height_ratio))

        k_start = int(l_diff_im * (length_ratio))
        k_end = int(l_img - l_diff_im * (1 - length_ratio))

        #print(f'{i_start:3}:{i_end:3}, {j_start:3}:{j_end:3}, {k_start:3}:{k_end:3}; {w_img:3} - {w_diff_im:3}, {h_img:3} -  {h_diff_im:3}, {l_img:3} - {l_diff_im:3}')
        print(i_end-i_start, j_end-j_start, k_end-k_start)

        new_imgs.append(img[i_start:i_end, j_start:j_end, k_start:k_end])
        # new_imgs.append(img[k_start:k_end, j_start:j_end, i_start:i_end])

    if find_dynamic_bounding_ratios:
        return new_imgs, dynamic_bounding_ratios
    else:
        return new_imgs


def resample_image_helper(arg):
    img, out_spacing, is_label = arg
    return resample_image(img, out_spacing, is_label)


def resample_image_standardize_helper(arg):
    img, out_size, is_label = arg
    return resample_image_standardize(img, out_size, is_label)


def resample_images(imgs, out_spacing=(1.0, 1.0, 1.0), is_label=False, n_jobs=mp.cpu_count()-1):
    args = []
    for img in imgs:
        args.append((img, out_spacing, is_label))

    if n_jobs == 1:
        results = []
        for arg in tqdm(args):
            results.append(resample_image_helper(arg))
        return results

    with mp.Pool(processes=n_jobs) as p:
        return list(tqdm(p.imap(resample_image_helper, args), total=len(args)))


def resample_images_standardize(imgs, out_size=(64, 64, 64), is_label=False, n_jobs=mp.cpu_count()-1):
    args = []
    for img in imgs:
        args.append((img, out_size, is_label))

    if n_jobs == 1:
        results = []
        for arg in tqdm(args):
            results.append(resample_image_standardize_helper(arg))
        return results

    with mp.Pool(processes=n_jobs) as p:
        return list(tqdm(p.imap(resample_image_standardize_helper, args), total=len(args)))

####################################################################################################

def respace_img(img_path_in, img_path_out, out_spacing, is_label):
    img = sitk.ReadImage(img_path_in)
    img = resample_image(img, out_spacing=out_spacing, is_label=is_label)
    sitk.WriteImage(img, img_path_out)
    new_size = img.GetSize()
    print('Respace out:', new_size, img_path_out)
    return (img_path_out, new_size)


# TODO: document sorting logic
def respace_directories(paths_in=[], paths_out=[], is_labels=[], out_spacing=(), n_jobs=1) -> dict:
    args = []
    for dir_path_in, dir_path_out, is_label in zip(paths_in, paths_out, is_labels):
        file_names_in = sorted(os.listdir(dir_path_in))

        for file_name_in in file_names_in:
            file_path_in = os.path.join(dir_path_in, file_name_in)
            file_path_out = os.path.join(dir_path_out, file_name_in)
            args.append((file_path_in, file_path_out, out_spacing, is_label))
    
    with mp.Pool(processes=n_jobs) as pool:
        path_size_pairs = list(pool.starmap(respace_img, args))

    # Gather results and sort them by filename
    result_dict = dict()
    for path in paths_out:
        result_dict[path] = ([], [])
    
    for path, size in path_size_pairs:
        img_dir = os.path.split(path)[0]
        img_name = os.path.split(path)[1]
        result_dict[img_dir][0].append(img_name)
        result_dict[img_dir][1].append(size)

    for dir, name_size_pairs in result_dict.items():
        name_size_pairs = list(zip(*name_size_pairs))
        name_size_pairs = sorted(name_size_pairs, key=lambda x: x[0])
        name_size_pairs = list(zip(*name_size_pairs))
        sizes = name_size_pairs[1]
        result_dict[dir] = list(sizes)
    
    return result_dict


def find_dynamic_cropping_ratios_for_img(img_path, crop_to):
    img = sitk.ReadImage(img_path)
    w_img, h_img, l_img = img.GetSize()

    w_diff_im = w_img - crop_to[0]
    h_diff_im = h_img - crop_to[1]
    l_diff_im = l_img - crop_to[2]

    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(sitk.OtsuThreshold(img, 0, 1))
    bounding_box = label_shape_filter.GetBoundingBox(1)

    i_start_max, j_start_max, k_start_max = bounding_box[:3]
    i_span_min, j_span_min, k_span_min = bounding_box[3:]

    i_end_min = i_start_max + i_span_min
    j_end_min = j_start_max + j_span_min
    k_end_min = k_start_max + k_span_min

    if (l_diff_im > l_img - k_span_min
        or h_diff_im > h_img - j_span_min
        or w_diff_im > w_img - i_span_min):
        print((f'Bounding box on {img_path} does not meet minimum size ' +
        'requirement for cropping; cropping will trespass minimum bounding box.'))

    i_end_min_from_end = w_img - i_end_min
    j_end_min_from_end = h_img - j_end_min
    k_end_min_from_end = l_img - k_end_min

    if i_start_max > i_end_min_from_end:
        a_i = i_end_min_from_end
        b_i = i_start_max
        width_ratio = max((a_i / b_i) / 2, 1 - (a_i / b_i) / 2)
    else:
        b_i = i_end_min_from_end
        a_i = i_start_max
        width_ratio = min((a_i / b_i) / 2, 1 - (a_i / b_i) / 2)

    if j_start_max > j_end_min_from_end:
        a_j = j_end_min_from_end
        b_j = j_start_max
        height_ratio = max((a_j / b_j) / 2, 1 - (a_j / b_j) / 2)
    else:
        b_j = j_end_min_from_end
        a_j = j_start_max
        height_ratio = min((a_j / b_j) / 2, 1 - (a_j / b_j) / 2)

    if k_start_max > k_end_min_from_end:
        a_k = k_end_min_from_end
        b_k = k_start_max
        length_ratio = max((a_k / b_k) / 2, 1 - (a_k / b_k) / 2)
    else:
        b_k = k_end_min_from_end
        a_k = k_start_max
        length_ratio = min((a_k / b_k) / 2, 1 - (a_k / b_k) / 2)
    
    return width_ratio, height_ratio, length_ratio


def find_dynamic_cropping_ratios_for_dir(dir_path_in, min_dims, n_jobs=1):
    file_names_in = sorted(os.listdir(dir_path_in))
    file_paths_in = [os.path.join(dir_path_in, name) for name in file_names_in]
    args = [(path, min_dims) for path in file_paths_in]

    with mp.Pool(processes=n_jobs) as pool:
        return pool.starmap(find_dynamic_cropping_ratios_for_img, args)


def crop_image(path_in, path_out, crop_to, cropping_ratio, remove_in):
    img = sitk.ReadImage(path_in)

    if remove_in:
        os.remove(path_in)
    
    width_ratio, height_ratio, length_ratio = cropping_ratio

    w_img, h_img, l_img = img.GetSize()
    w_diff_im = w_img - crop_to[0]
    h_diff_im = h_img - crop_to[1]
    l_diff_im = l_img - crop_to[2]
    
    i_start = int(w_diff_im * width_ratio)
    i_end = int(w_img - w_diff_im * (1 - width_ratio))

    j_start = int(h_diff_im * height_ratio)
    j_end = int(h_img - h_diff_im * (1 - height_ratio))

    k_start = int(l_diff_im * (length_ratio))
    k_end = int(l_img - l_diff_im * (1 - length_ratio))

    img = img[i_start:i_end, j_start:j_end, k_start:k_end]
    print('Crop out:', img.GetSize(), path_out)

    sitk.WriteImage(img, path_out)


def crop_directories(paths_in=[], paths_out=[], crop_to=(), cropping_ratios=(), inplace=False, n_jobs=1):
    args = []
    if inplace:
        paths_out = paths_in
    elif paths_out == []:
        paths_out = paths_in
        inplace = True

    for i, (dir_path_in, dir_path_out) in enumerate(zip(paths_in, paths_out)):
        file_names_in = sorted(os.listdir(dir_path_in))

        for file_name_in in file_names_in:
            path_in = os.path.join(dir_path_in, file_name_in)
            path_out = os.path.join(dir_path_out, file_name_in)
            cropping_ratio = cropping_ratios[i]
            arg = (path_in, path_out, crop_to, cropping_ratio, inplace)
            args.append(arg)
    
    with mp.Pool(processes=n_jobs) as pool:
        pool.starmap_async(crop_image, args).get()


def resize_image(path_in, path_out, out_size, is_label, remove_in):
    img = sitk.ReadImage(path_in)
    if remove_in:
        os.remove(path_in)

    img = resample_image_standardize(img, out_size=out_size, is_label=is_label)

    print('Resize out:', img.GetSize(), path_out)
    sitk.WriteImage(img, path_out)


def resize_directories(paths_in=[], paths_out=[], is_labels=[], out_size=(), inplace=False, n_jobs=1):
    args = []
    if inplace:
        paths_out = paths_in
    elif paths_out==[]:
        paths_out = paths_in
        inplace = True

    for dir_path_in, dir_path_out, is_label in zip(paths_in, paths_out, is_labels):
        file_names_in = sorted(os.listdir(dir_path_in))

        for file_name_in in file_names_in:
            file_path_in = os.path.join(dir_path_in, file_name_in)
            file_path_out = os.path.join(dir_path_out, file_name_in)
            args.append((file_path_in, file_path_out, out_size, is_label, inplace))
    
    with mp.Pool(processes=n_jobs) as pool:
        pool.starmap_async(resize_image, args).get()


def augmentation_pipeline(img_path, msk_path, img_path_out, msk_path_out,
                          out_spacing=(1, 1, 1), out_size=(64, 64, 64),
                          n_jobs=1):

    try:
        mp.set_start_method('spawn')
    except NameError:
        pass

    img_path = os.path.abspath(img_path)
    msk_path = os.path.abspath(msk_path)
    img_path_out = os.path.abspath(img_path_out)
    msk_path_out = os.path.abspath(msk_path_out)

    sizings = respace_directories(
        paths_in=(img_path, msk_path),
        paths_out=(img_path_out, msk_path_out),
        is_labels=[False, True],
        out_spacing=out_spacing,
        n_jobs=n_jobs)
    
    msk_sizes = np.array(sizings[msk_path_out])
    min_dims = msk_sizes.min(axis=0)

    cropping_ratios = find_dynamic_cropping_ratios_for_dir(msk_path_out, min_dims)

    crop_directories(
        paths_in=(img_path_out, msk_path_out),
        crop_to=min_dims,
        cropping_ratios=cropping_ratios,
        inplace=True,
        n_jobs=n_jobs)

    resize_directories(
        paths_in=(img_path_out, msk_path_out),
        is_labels=[False, True],
        out_size=out_size,
        inplace=True,
        n_jobs=n_jobs
    )

def validate_augmentation(hi_res_images_path, hi_res_masks_path, lo_res_images_path, lo_res_masks_path):
    lo_res_path = os.path.split(lo_res_images_path)[0]
    hi_res_path = os.path.split(hi_res_images_path)[0]

    ### LOOK FOR MASKS AND IMAGE FILES
    masks_list = os.listdir(lo_res_masks_path)
    images_list = os.listdir(lo_res_images_path)

    ############## VERIFY DATA INTEGRITY ###############
    num_masks = len(masks_list)
    num_images = len(images_list) 

    print('Testing data integrity...', end='')

    ### 1. VERIFY MATCHING NUMBER OF MASKS AND IMAGES
    if num_masks != num_images:
        raise FileNotFoundError(f"Unequal number of masks and images in {lo_res_path}")

    masks_list.sort()
    images_list.sort()

    image_mask_pairs = list(zip(images_list, masks_list))

    ### 2. VERIFY 1-1 CORRESPONDENCE BETWEEN MASKS AND IMAGES
    for pair in image_mask_pairs:
        name_1 = ''.join(pair[1].split('.')[0].split('-')[1:])
        name_2 = ''.join(pair[0].split('.')[0].split('-')[1:])
        if (name_1 != name_2):
            msg = f'Incomplete correspondence between masks and images in {lo_res_path}: '
            msg += f'found non-matching (mask, image) pair {name_1, name_2} from files {pair}.'
            raise FileNotFoundError(msg)
    
    ### 3. VERIFY 1-1 CORRESPONDENCE BETWEEN HI-RES AND LO-RES
    hi_res_pairs = list(zip(sorted(os.listdir(hi_res_masks_path)), sorted(os.listdir(hi_res_images_path))))
    
    for hi_res_pair, lo_res_pair in zip(hi_res_pairs, image_mask_pairs):
        lo_res_name_1 = ''.join(lo_res_pair[0].split('.')[0].split('-')[1:])
        lo_res_name_2 = ''.join(lo_res_pair[1].split('.')[0].split('-')[1:])
        hi_res_name_1 = ''.join(hi_res_pair[0].split('.')[0].split('-')[1:])
        hi_res_name_2 = ''.join(hi_res_pair[1].split('.')[0].split('-')[1:])
        
        if lo_res_name_1 != hi_res_name_1 or lo_res_name_2 != hi_res_name_2:
            msg = f'Incomplete correspondence between hi-res and lo-res in {lo_res_path} and {hi_res_path}. '
            msg += f'Hi-res pair {hi_res_pair} does not match {lo_res_pair}. Did you finish resampling?'
            raise FileNotFoundError(msg)

    print('Passed!')
    
    ################# VERIFY LABEL QUANTIZATION ##################
    
    print('Testing label quantization...', end='')

    expected_quantization = [0,1]
    
    for mask in os.listdir(lo_res_masks_path):
        lo_res_mask_path = os.path.join(lo_res_masks_path, mask)
        sample_mask_lo_res = sitk.ReadImage(lo_res_mask_path)
        unique_values = list(pd.Series(sitk.GetArrayFromImage(sample_mask_lo_res).ravel()).unique())
        if unique_values != expected_quantization:
            raise ValueError('Masks arrays not quantized to the set {0, 1}. Did you pass is_label=True?')
    else:
        print(f'Passed! Quantized to {expected_quantization}')

# def resampling_pipeline(imgs, masks, out_spacing=(1.0, 1.0, 1.0),
#                         out_size=(64, 64, 64), masks_only=False, n_jobs=1):
#     """Resamples images to common voxel spacing, crops to common dimensions,
#     then resamples to 64x64x64 size, in that order.

#     Arguments:
#         imgs {[type]} -- [description]
#         masks {[type]} -- [description]

#     Keyword Arguments:
#         out_spacing {tuple} -- [description] (default: {(1.0, 1.0, 1.0)})
#         out_size {tuple} -- [description] (default: {(64, 64, 64)})

#     Returns:
#         tuple -- A two-tuple of image and mask lists
#     """

#     print('Resampling mask spacing')
#     masks = resample_images(masks, out_spacing=out_spacing, is_label=True, n_jobs=n_jobs)
#     print('Cropping masks')
#     masks, bounding_ratios = crop_images_to_common_dimensions(masks, find_dynamic_bounding_ratios=True)
#     print('Resampling mask size')
#     masks = resample_images_standardize(masks, out_size=out_size, is_label=True, n_jobs=n_jobs)
    
#     if masks_only:
#         return masks
    
#     print('Resampling image spacing')
#     imgs = resample_images(imgs, out_spacing=out_spacing, is_label=False, n_jobs=n_jobs)
#     print('Cropping images')
#     imgs = crop_images_to_common_dimensions(imgs, dynamic_bounding_ratios=bounding_ratios)
#     print('Resampling image size')
#     imgs = resample_images_standardize(imgs, out_size=out_size, is_label=False, n_jobs=n_jobs)

#     return imgs, masks

def k3d_plot(img, color_range=None):

    import k3d
    import numpy as np
        
    if color_range is None:
        img_array = sitk.GetArrayFromImage(img).ravel()
        color_range = [img_array.min(), img_array.max()]

    im_sitk = img
    img  = sitk.GetArrayFromImage(im_sitk)
    size = np.array(im_sitk.GetSize()) * np.array(im_sitk.GetSpacing())
    im_sitk.GetSize()

    volume = k3d.volume(
        img.astype(np.float32),
        alpha_coef=1000,
        shadow='dynamic',
        samples=1200,
        shadow_res=64,
        shadow_delay=50,
        color_range=color_range,
        color_map=(np.array(k3d.colormaps.matplotlib_color_maps.Gist_heat).reshape(-1,4)
                   * np.array([1,1.75,1.75,1.75])).astype(np.float32),
        compression_level=9
    )

    volume.transform.bounds = [-size[0]/2,size[0]/2,
                               -size[1]/2,size[1]/2,
                               -size[2]/2,size[2]/2]

    plot = k3d.plot(camera_auto_fit=False)
    plot += volume
    plot.lighting = 2
    plot.display()


def animate_2d_plot(sitk_image, repeat=True):
    import matplotlib.pyplot as plt
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    import matplotlib.animation as animation

    fig = plt.figure()
    imgs = []

    shape = sitk_image.GetSize()
    for m in range(0, 3):
        
        step = shape[m] // 8
        for l in range(0, shape[m], step):
            if m == 2:
                i = l
                j = k = 0
                img = plt.imshow(sitk.GetArrayFromImage(sitk_image[i,:,:]), animated=True)
                text = f'Y vs. Z @ x = {i:3n}  [{i:2n}, :, :]'
            elif m == 1:
                j = l
                i = k = 0
                img = plt.imshow(sitk.GetArrayFromImage(sitk_image[:,j,:]), animated=True)
                text = f'X vs. Z @ y = {j:3n}  [:, {j:2n}, :]'
            elif m == 0:
                k = l
                i = j = 0
                img = plt.imshow(sitk.GetArrayFromImage(sitk_image[:,:,k]), animated=True)
                text = f'X vs. Y @ z = {k:3n}  [:, :, {k:2n}]'

            #img = plt.imshow(sitk.GetArrayFromImage(sample_mask[i,:,:]), animated=True)
            #j = k = 0
            txt = plt.text(10, -3, f'{text}')
            imgs.append([img, txt])

    ani = animation.ArtistAnimation(fig, imgs, interval=100, blit=True,
                                    repeat_delay=1000, repeat=repeat)

    plt.close()
    return ani

def imshow_sitk(img):
    import matplotlib.pyplot as plt
    return plt.imshow(sitk.GetArrayFromImage(img))

def interact_sitk_helper(img, x0, x1, y0, y1, x):
    shape = img.GetSize()
    
    x1 = min(shape[0], x1)
    y1 = min(shape[1], y1)

    imshow_sitk(img[x0:x1,y0:y1,x])

def interact_sitk(image):
    from ipywidgets import interact, fixed
    shape = image.GetSize()
    interact(interact_sitk_helper, x0 = (0, shape[0]//2), x1 = (shape[0]//2, shape[0]), y0 = (0, shape[1]//2), y1 = (shape[1]//2, shape[1]), img = fixed(image), x = (0, shape[2]-1))