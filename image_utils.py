import json
import os

import SimpleITK as sitk
import nibabel as nib
import numpy as np
from PIL import Image


def read_image(imgpath):
    """Read image and return numpy matrix. Possible image types: .mhd, .mha, .nii, .nii.gz and standard types like .png and .jpeg.
    :param imgpath: path to read image from
    :return: numpy image
    """
    if imgpath.endswith('nii') or imgpath.endswith('nii.gz'):
        image_file = nib.load(imgpath)
        image_file_np = np.array(image_file.get_fdata())
    elif imgpath.endswith('mha') or imgpath.endswith('mhd') or imgpath.endswith('raw'):
        image_file = sitk.ReadImage(imgpath)
        image_file_np = sitk.GetArrayFromImage(image_file)
    else:
        try:
            image_file = Image.open(imgpath)
            image_file_np = np.array(image_file)
        except:
            print(
                'Image Type not recognized! Currently supporting .mhd, .mha, .nii, .nii.gz and standard types like .png and .jpeg')
            return None
    return image_file_np


def save_image(img_np, imgpath, affine=np.eye(4)):
    """Save numpy image matrix to file. Possible image types: .mhd, .mha, .nii, .nii.gz and standard types like .png and .jpeg.
    :param img_np: numpy image
    :param imgpath: path to save image to
    :param affine: if nii file type is chose, affine matrix (4x4) can be defined"""
    if imgpath.endswith('nii') or imgpath.endswith('nii.gz'):
        nib.save(nib.Nifti1Image(img_np, affine), imgpath)
    elif imgpath.endswith('mha') or imgpath.endswith('mhd') or imgpath.endswith('raw'):
        sitk.WriteImage(sitk.GetImageFromArray(img_np, isVector=False), imgpath)
    else:
        try:
            image_file = Image.fromarray(img_np)
            image_file.save(imgpath)
        except:
            print(
                'Image Type not recognized! Currently supporting .mhd, .mha, .nii, .nii.gz and standard types like .png and .jpeg')
            return None


def normalize_image(img, min_value=0, max_value=255, value_type='uint8'):
    """Normalize image values in a certain range.
    :param img: input numpy image
    :param min_value: minimum gray value
    :param max_value: maximum gray value
    :param value_type: type of normalized img
    :return: nomralized image
    """
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = img * (max_value - min_value) + min_value
    return img.astype(value_type)


def crop(out_dir, img_nr, image_np, mask_np=None):
    bb_coords = np.where(image_np > 0)
    offset = 3
    x_min = np.min(bb_coords[0])
    x_min = max(x_min - offset, 0)
    x_max = np.max(bb_coords[0])
    x_max = min(x_max + offset, image_np.shape[0])
    y_min = np.min(bb_coords[1])
    y_min = max(y_min - offset, 0)
    y_max = np.max(bb_coords[1])
    y_max = min(y_max + offset, image_np.shape[1])
    z_min = np.min(bb_coords[2])
    z_min = max(z_min - offset, 0)
    z_max = np.max(bb_coords[2])
    z_max = min(z_max + offset, image_np.shape[2])

    info_path = os.path.join(out_dir, 'shape_info_' + str(img_nr) + '.json')
    info_json = {'pat': '.', 'imgSizeWhole': image_np.shape,
                 'imgSizeCropped': image_np[x_min:x_max, y_min:y_max, z_min:z_max].shape, 'affine': None,
                 'coords': []}
    info_json['coords'].append({'x_min': int(x_min), 'y_min': int(y_min), 'z_min': int(z_min)})
    with open(info_path, 'w') as outfile:
        json.dump(info_json, outfile)
    # crop
    image_np = image_np[x_min:x_max, y_min:y_max, z_min:z_max]
    if mask_np is not None:
        mask_np = mask_np[x_min:x_max, y_min:y_max, z_min:z_max]
    return image_np, mask_np
