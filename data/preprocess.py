import os
import random

import numpy as np
from image_utils import save_image, normalize_image, crop
from data.sketch import generate_sketch


def preprocess(dataset, output_dir, valid_pr, out_img_list, out_img_list_valid, out_sketch_list, out_sketch_list_valid):
    random.seed(2017)
    img_nr = 0
    for sample in dataset:
        img_nr += 1
        image = sample.get('image')
        mask = sample.get('mask')
        image_np = np.flip(np.flip(np.transpose(image, (2, 1, 0)), 0), 1)
        mask_np = np.flip(np.flip(np.transpose(mask, (2, 1, 0)), 0), 1) if mask is not None else None
        cropped_image, cropped_mask = crop(output_dir, img_nr, image_np, mask_np)
        normalized_image = normalize_image(cropped_image)
        image = normalized_image
        mask = cropped_mask

        image_path = os.path.join(output_dir, 'crossmoda_' + str(img_nr) + '_0000.nii')
        save_image(image, image_path)
        if mask is not None:
            mask_path = os.path.join(output_dir, 'crossmoda_' + str(img_nr) + '.nii')
            save_image(mask, mask_path)


        #### Sketch
        sketch_img = generate_sketch(image_np)
        # combine with tumor mask
        if mask is not None:
            mask_np = np.maximum(np.zeros_like(mask_np), (64 * mask_np.astype('int32') - 1))
            sketch_img = np.maximum(sketch_img, mask_np.resize(np.shape(sketch_img)))
        else:
            sketch_img = sketch_img

        sketch_path = os.path.join(output_dir, 'sketch_' + str(img_nr) + '.nii')
        save_image(sketch_img.astype('uint8'), sketch_path)

        # img for validation?
        if random.random() <= valid_pr:
            out_img_list_valid.write(image_path + '\n')
            out_sketch_list_valid.write(sketch_path + '\n')
        else:
            out_img_list.write(image_path + '\n')
            out_sketch_list.write(sketch_path + '\n')
