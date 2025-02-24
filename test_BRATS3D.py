from __future__ import print_function

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from coords3D_utils import crop_img
from coords3D_utils import get_all_coords
from data.dataset import MedicalImageDataset3D
from image_utils import save_image
from networks3D_HR import define_G
from networks3D_LR import define_G_LR

# Training settings
parser = argparse.ArgumentParser(description='Test: Create high resolution images')
parser.add_argument('--data_root', type=str, default='../Data',
                    help='root directory of the dataset')
parser.add_argument('--results_root', type=str, default='../Results',
                    help='root directory of results (tests and  checkpoints)')
parser.add_argument('--experiment_type', type=str, default='SKETCH2BRATST23D',
                    help='name of experiment (also name of folders)')
parser.add_argument('--LR_size', type=int, default=32, nargs='+', help='LR image size (isotrop) (scale 0)')
parser.add_argument('--HR_size', type=int, default=64, nargs='+',  help='HR image size (isotrop) (last scale)')
parser.add_argument('--patch_size', type=int, default=32, nargs='+', help='Size of patches')
parser.add_argument('--batch_size', type=int, default=32, help='Number patches to be loaded at once')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--device', type=int, default=0, help="Device nr (def: 0)")

opt = parser.parse_args()
HR_size = tuple(opt.HR_size)
LR_size = tuple(opt.LR_size)
patch_size = tuple(opt.patch_size)

print(opt)
device = torch.device('cuda', opt.device)

with torch.cuda.device(opt.device):
    print('===> Loading datasets:', HR_size)
    input_dir = os.path.join(opt.data_root, opt.experiment_type)
    output_dir = os.path.join(opt.results_root, opt.experiment_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    test_output_dir = os.path.join(output_dir, 'LPBATest')
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    # load test dataset
    HR_test_set = MedicalImageDataset3D(input_dir, mode="test", listname='data_list.txt',
                                        new_size_imgs=HR_size, new_size_edges=HR_size)
    HR_test_data_loader = DataLoader(HR_test_set,
                                     batch_size=1, shuffle=False,
                                     num_workers=8, pin_memory=False)

    print('===> Building model')
    LRG = define_G_LR(1, 1, norm='instance', scale=True).to(device)  # low-resolution generator
    LRG.load_state_dict(torch.load(os.path.join(output_dir + '_LR', 'NETG_best.pth'),
                                   map_location=lambda storage, loc: storage))
    LRG = LRG.eval()
    criterionMSE = nn.MSELoss()
    # filenames for saving images
    fileNames = open(os.path.join(input_dir, 'A/%s' % 'test/data_list.txt')).readlines()


def stitch_image(out_img_size, patches, coords):
    """Stitch image out of patches on certain coordinates.
    :param out_img_size: size of output image
    :param patches: a batch of patches
    :param coords: coordinates of the patches
    :return: stiched image
    """
    stitched_image = np.zeros(out_img_size)
    norm_image = np.zeros(out_img_size)  # normalize overlaps
    for patch_i in range(0, coords.shape[0]):
        ss = coords[patch_i, :, 0]  # slice start
        se = coords[patch_i, :, 1] + 1  # slice end
        print("ss:", ss, "se:", se, "shape:", np.shape(np.ones(se - ss)), np.shape(norm_image[ss[0]:se[0], ss[1]:se[1], ss[2]:se[2]]), np.shape(ss), np.shape(se), np.shape(se-ss), se-ss)
        norm_image[ss[0]:se[0], ss[1]:se[1], ss[2]:se[2]] = norm_image[ss[0]:se[0], ss[1]:se[1], ss[2]:se[2]] + np.ones(
            se - ss)
        stitched_image[ss[0]:se[0], ss[1]:se[1], ss[2]:se[2]] = stitched_image[ss[0]:se[0], ss[1]:se[1],
                                                                ss[2]:se[2]] + patches[patch_i, 0, :, :, :]
    norm_image[np.where(norm_image <= 0)] = 1
    return stitched_image / norm_image


def LR_to_HR(HR_size, LRI_fake, HRE, G, reception_field=20, part_size=20):
    """Function to turn a low-resolution image into a higher resolution image, given a train GAN.
    :param HR_size: size of high-resolution images
    :param LRI_fake: fake (GAN generated) low-resolution image
    :param HRE: high resolution edge image
    :param G: generator of a GAN
    :param reception_field: size of reception field on patches (size without padding artifacts)
    :param part_size: to avoid using much GPU memory, we partition the patches into batches of size part_size
    :return stitched_HR_fake: stitched fake HR image

    """
    offset0 = int((patch_size[0]) / 2) - int(math.floor(reception_field / 2))
    offset1 = int((patch_size[1]) / 2) - int(math.floor(reception_field / 2))
    offset2 = int((patch_size[2]) / 2) - int(math.floor(reception_field / 2))
    offset = (offset0, offset1, offset2)
    # sample all patches of image
    coords_small = get_all_coords((reception_field, reception_field, reception_field),
                                  patch_size, HR_size, 1)
    print('LRI_fake.shape:', LRI_fake.shape)
    scale_factor = (HR_size[0] / LRI_fake.shape[2], HR_size[1] / LRI_fake.shape[3], HR_size[2] / LRI_fake.shape[4])
    new_patch_size = (patch_size[0] / scale_factor[0], patch_size[1] / scale_factor[1], patch_size[2] / scale_factor[2])
    padding0 = int(round((patch_size[0] - new_patch_size[0]) / 2.))
    padding1 = int(round((patch_size[1] - new_patch_size[1]) / 2.))
    padding2 = int(round((patch_size[2] - new_patch_size[2]) / 2))
    padding = ((padding0, padding0), (padding1, padding1), (padding2, padding2))
    print("padding:", padding, patch_size, new_patch_size, "scale_factor:", scale_factor)
    LRI_fake_data = LRI_fake.data.cpu().numpy()[0, 0, :, :, :]
    print('LRI_fake_data:', np.shape(LRI_fake_data))
    LR_img_padded = np.pad(LRI_fake_data, padding, 'constant', constant_values=-1)
    coords_big = np.zeros_like(coords_small)
    coords_big[:, :, 0] = np.floor(coords_small[:, :, 0] / scale_factor)
    coords_big[:, :, 1] = np.floor(coords_small[:, :, 0] / scale_factor) + opt.patch_size - 1

    LRP_fake = crop_img(coords_big, np.expand_dims(LR_img_padded, 0), patch_size)
    HREP = crop_img(coords_small, HRE.data.cpu().numpy(), patch_size)

    LRP_fake = torch.from_numpy(LRP_fake).to(device)
    HREP = torch.from_numpy(HREP).to(device)

    if part_size > 1:
        n_parts = int(HREP.shape[0] / part_size)
    predictions = []
    print("n_parts:", n_parts)
    # memory efficient
    for p in range(0, n_parts + 1):
        print('Partition: ' + str(p))
        if p < n_parts:
            current_size = part_size
        else:
            current_size = HREP.shape[0] - p * part_size
        if current_size <= 0:
            continue
        small_batch = (LRP_fake[p * part_size:p * part_size + current_size, :, :, :].detach(),
                       HREP[p * part_size:p * part_size + current_size, :, :, :].detach())

        HRP_fake = G(small_batch)  # generate patches
        predictions.append(HRP_fake.cpu().data)
        del HRP_fake
        del small_batch
    prediction = torch.cat(predictions, 0)
    prediction_np = prediction.data.cpu().numpy()

    stitched_HR_fake_no_rf = stitch_image(HR_size, prediction_np,
                                          coords_small)  # without reception field, only used for padding

    # coordinates with reception field
    coords_small[:, :, 0] = coords_small[:, :, 0] + offset
    coords_small[:, :, 1] = coords_small[:, :, 1] - offset
    coords_np = coords_small
    ss = offset
    se = (offset0 + reception_field, offset1 + reception_field, offset2 + reception_field)

    prediction_np = prediction_np[:, :, ss[0]:se[0], ss[1]:se[1], ss[2]:se[2]]  # with reception field
    stitched_HR_fake = stitch_image(HR_size, prediction_np, coords_np)
    # kind of padding with so the edges have the padding artifacts without using smaller reception field
    stitched_HR_fake[0:offset0, :, :] = stitched_HR_fake_no_rf[0:offset0, :, :]
    stitched_HR_fake[:, 0:offset1, :] = stitched_HR_fake_no_rf[:, 0:offset1, :]
    stitched_HR_fake[:, :, 0:offset2] = stitched_HR_fake_no_rf[:, :, 0:offset2]
    stitched_HR_fake[-offset0:, :, :] = stitched_HR_fake_no_rf[-offset0:, :, :]
    stitched_HR_fake[:, -offset1:, :] = stitched_HR_fake_no_rf[:, -offset1:, :]
    stitched_HR_fake[:, :, -offset2:] = stitched_HR_fake_no_rf[:, :, -offset2:]
    return stitched_HR_fake


# the real deal
with torch.cuda.device(opt.device):
    part_size = opt.batch_size
    reception_field = 20

    for i, batch in enumerate(HR_test_data_loader):
        currentFilename = os.path.basename(fileNames[i]).replace('sketch_', '')
        currentFilename = currentFilename.replace('\n', '')
        print(currentFilename)
        currentPatNr = currentFilename.split('.')[0]
        HRE_orig = batch['A']
        # HRI_orig = batch['B']

        # generate scale 0
        print("orig shape:", np.shape(HRE_orig))
        LRE = nn.functional.interpolate(HRE_orig, (2 * LR_size[0], 2 * LR_size[1], 2 * LR_size[2]), mode='trilinear',
                                        align_corners=True).float().squeeze_(0)
        LRI_fake = LRG(LRE.unsqueeze_(0).to(device))
        print("info:", np.shape(LRI_fake), LR_size)
        LRI_fake = nn.functional.interpolate(LRI_fake, LR_size, mode='trilinear',
                                             align_corners=True).float()
        LRI_fake = LRI_fake.data.cpu().numpy()[0, 0, :, :, :]
        l = 0  # level
        currSize = LR_size

        save_image(127.5 * (LRI_fake + 1), test_output_dir + '/fake_level' + str(0) + '_%s' % currentFilename + '.gz')

        # generate further scales
        print("emmmm:", currSize, HR_size)
        while currSize < HR_size:
            l += 1
            currSize = (currSize[0] * 2, currSize[1] * 2, currSize[2] * 2)
            HRE = nn.functional.interpolate(HRE_orig, currSize, mode='trilinear',
                                            align_corners=True).float().squeeze(0)
            # HRI_real = nn.functional.interpolate(HRI_orig, (currSize, currSize, currSize), mode='trilinear',
            # align_corners=True).float().squeeze(0)
            # build hr generator network
            HRG = define_G(2, 1, 64, norm='instance').to(device)
            ckp_path = os.path.join(output_dir + '_HR' + str(l), 'NETG_best.pth')
            HRG.load_state_dict(torch.load(ckp_path, map_location=lambda storage, loc: storage))
            HRG.eval()

            LRI_fake = torch.from_numpy(LRI_fake).unsqueeze(0).unsqueeze(0)
            HRI_fake = LR_to_HR(currSize, LRI_fake, HRE, HRG, reception_field, part_size=part_size)  # a scale higher
            save_image(127.5 * (HRI_fake + 1).astype('int16'),
                       test_output_dir + '/fake_level' + str(l) + '_%s' % currentFilename + '.gz')
            LRI_fake = HRI_fake
            del HRG
