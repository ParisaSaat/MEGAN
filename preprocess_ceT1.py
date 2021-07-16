import argparse
import os

from data.dataset import VS
from data.preprocess import preprocess

"""Script for preprocessig the BRATS training data. Extractzing edges, cropping images, svaing list files."""

parser = argparse.ArgumentParser(description='Preprocess BRATS data')
parser.add_argument('--in_img_dir', type=str, default='../Data/ceT1_Training',
                    help='Image dir of the ceT1 images')
parser.add_argument('--mask_dir', type=str, default='../Data/BRATS2015_Training',
                    help='Image dir of the BRATS images')
parser.add_argument('--out_img_dir', type=str, default='../Data/BRATST2_3D',
                    help='Output dir of preprocessed images and sketches')
parser.add_argument('--out_list_dir', type=str, default='../Data/SKETCH2BRATST23D',
                    help='Output dir of images and sketches lists (training and validation)')
parser.add_argument('--valid_pr', type=float, default=0.1, help='[0,1] percentage of images for validation')

if __name__ == '__main__':
    opt = parser.parse_args()

    ###python MEGAN/preprocess_ceT1.py --in_img_dir /imagedata/BRATS2016/BRATS2015_Training --out_img_dir /share/data_tiffy2/uzunova/GAN/Data/BRATST1_3DTmp --out_list_dir /share/data_tiffy2/uzunova/GAN/Data/SKETCH2BRATST13D/

    img_dir = opt.in_img_dir
    mask_dir = opt.mask_dir
    out_dir = opt.out_img_dir
    out_list_dir = opt.out_list_dir
    sketch_list_dir = os.path.join(out_list_dir, 'A/train')
    img_list_dir = os.path.join(out_list_dir, 'B/train')

    valid_pr = opt.valid_pr
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(sketch_list_dir):
        os.makedirs(sketch_list_dir)
    if not os.path.exists(img_list_dir):
        os.makedirs(img_list_dir)

    out_img_list = open(img_list_dir + '/data_list.txt', 'w')  # training img list
    out_sketch_list = open(sketch_list_dir + '/data_list.txt', 'w')  # training sketches list

    out_img_list_valid = open(img_list_dir + '/data_list_valid.txt', 'w')  # validation img list
    out_sketch_list_valid = open(sketch_list_dir + '/data_list_valid.txt', 'w')  # validation sketch list

    data = VS(img_dir, mask_dir)
    preprocess(data, output_dir=out_dir, valid_pr=valid_pr,
               out_img_list=out_img_list, out_img_list_valid=out_img_list_valid, out_sketch_list=out_sketch_list,
               out_sketch_list_valid=out_sketch_list_valid)
