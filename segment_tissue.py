
from skimage.transform import pyramid_gaussian
from skimage import data
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square, remove_small_objects, binary_opening, remove_small_holes, binary_erosion, binary_dilation
from skimage import measure
from skimage import data
from skimage.morphology import skeletonize, disk, ball
from skimage.filters import threshold_local, threshold_sauvola, threshold_niblack, threshold_otsu, rank

from scipy import ndimage
from scipy import stats
import os
import numpy as np
import sys
import statistics
from scipy import ndimage
import math
import SimpleITK as sitk
import gc
from PIL import Image


# def GetStroma(volume, thresh = 0.3, alpha = 1, alpha_1d = 1, adaptive = False):

#     if adaptive:
#         print("running adaptive thresholding")
#         offset = 0
#         stroma = np.zeros(volume.shape, dtype=int)
#         radius = 101
#         selem = disk(radius)
        
#         for i in range(0, volume.shape[0]):
#             volume_itk = sitk.GetImageFromArray(volume[i,:,:])
#             adaptive_thresh = threshold_local(volume[i,:,:], radius, offset=offset)
#             stroma[i,:,:] = volume[i,:,:] > adaptive_thresh
#     else:
#         print("running simple thresholding")
#         stroma = volume > thresh

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def Run(file_name, output_file_name, threshold, alpha=1, alpha_1d = 1, min_size_tissue = 1, min_size_holes = 1):
    print(threshold)
    print(alpha)
    print(alpha_1d)
    
    volume_itk = sitk.ReadImage(file_name, imageIO="MetaImageIO")
    
    volume = sitk.GetArrayFromImage(volume_itk)
    
    tissue = volume > threshold
    
    if alpha > 0:
        for i in range(0, tissue.shape[0]):
            print("{}/{}".format(i, tissue.shape[0]))
            tissue[i,:,:] = ndimage.gaussian_filter(tissue[i,:,:], alpha)
    
    if alpha_1d > 0:
        tissue = ndimage.gaussian_filter1d(tissue, alpha_1d, 0)
    
    tissue = tissue > 0.5
    
    tissue = remove_small_objects(tissue, min_size_tissue)
    tissue = remove_small_holes(tissue, min_size_holes)
    tissue = tissue.astype(np.uint8)
    
    
    #tissue = getLargestCC(tissue)
    
    pos = int(tissue.shape[0]/2)
    im = Image.fromarray(tissue[pos,:,:]*50)
    im.save(file_name[:-4]+"_mask.png")

    im = Image.fromarray(volume[pos,:,:]*0.1)
    im.save(file_name[:-4]+".png")

    tissue_itk = sitk.GetImageFromArray(tissue)
    tissue_itk.SetSpacing(tissue_itk.GetSpacing())
    sitk.WriteImage(tissue_itk, output_file_name)

def SegmentStroma():
    gc.collect()

    Run('S:/Tristan/ForPaper/20190911_PDX_STG316_gfp_100x15um.mha', 'S:/Tristan/ForPaper/20190911_PDX_STG316_gfp_100x15um_mask.mha', 0.2, 5, 2, 1, 1, 0)
    return
    Run('S:/Tristan/Upsampled/20201109_PDX_STG1394_GFP_100x15um.mha', 'S:/Tristan/Mask/20201109_PDX_STG1394_GFP_100x15um_mask.mha', 0.15, 1, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20190911_PDX_STG316_gfp_100x15um.mha', 'S:/Tristan/Mask/20190911_PDX_STG316_gfp_100x15um_mask.mha', 0.2, 5, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20190916_PDX_STG316_gfp_100x15um_set2.mha', 'S:/Tristan/Mask/20190916_PDX_STG316_gfp_100x15um_set2_mask.mha', 0.2, 5, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20190919_PDX_STG139645_gfp_100x15um.mha', 'S:/Tristan/Mask/20190919_PDX_STG139645_gfp_100x15um_mask.mha', 0.2, 2.5, 2, 1, 0, 0)
    Run('S:/Tristan/Upsampled/20191021_PDX_STG143_100x15um.mha', 'S:/Tristan/Mask/20191021_PDX_STG143_100x15um_mask.mha', 0, 0.5, 2, 0, 0, 0)
    Run('S:/Tristan/Upsampled/20191209_PDX_STG143SC_50x15um.mha', 'S:/Tristan/Mask/20191209_PDX_STG143SC_50x15um_mask.mha', 0.1, 2, 2, 0, 0, 0)
    Run('S:/Tristan/Upsampled/20191210_PDX_STG143SC_100x15um_set2.mha', 'S:/Tristan/Mask/20191210_PDX_STG143SC_100x15um_set2_mask.mha', 0.1, 1, 2, 1, 0, 0)
    Run('S:/Tristan/Upsampled/20200219_PDX_AB559_50X15um.mha', 'S:/Tristan/Mask/20200219_PDX_AB559_50X15um_mask.mha', 0.3, 5, 2, 1, 0, 0)
    Run('S:/Tristan/Upsampled/20200225_PDX_Ab559_set2_100X15um.mha', 'S:/Tristan/Mask/20200225_PDX_Ab559_set2_100X15um_mask.mha', 0.1, 6, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20200226_PDX_Ab580_HC_gfp_100x15um.mha', 'S:/Tristan/Mask/20200226_PDX_Ab580_HC_gfp_100x15um_mask.mha', 0.15, 1.1, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20200227_PDX_Ab580_HC_gfp_set2_100x15um.mha', 'S:/Tristan/Mask/20200227_PDX_Ab580_HC_gfp_set2_100x15um_mask.mha', 0.2, 2, 2, 1, 0, 0)
    Run('S:/Tristan/Upsampled/20200615_PDX_AB580_GFP_100x15um_set1.mha', 'S:/Tristan/Mask/20200615_PDX_AB580_GFP_100x15um_set1_mask.mha', 0.1, 1.5, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20200629_PDX_PAR1040_GFP_100x15um_set1.mha', 'S:/Tristan/Mask/20200629_PDX_PAR1040_GFP_100x15um_set1_mask.mha', 0.1, 3, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20200707_PDX_PAR1040_GFP_100x15um_set2.mha', 'S:/Tristan/Mask/20200707_PDX_PAR1040_GFP_100x15um_set2_mask.mha', 0.1, 1.5, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20200804_PDX_AB580SC_gfp_100x15um_set1.mha', 'S:/Tristan/Mask/20200804_PDX_AB580SC_gfp_100x15um_set1_mask.mha', 0.1, 3, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20200817_PDX_AB764_017376_gfp_100x15um_set1.mha', 'S:/Tristan/Mask/20200817_PDX_AB764_017376_gfp_100x15um_set1_mask.mha', 0.05, 3, 2, 1, 0, 0)
    Run('S:/Tristan/Upsampled/20200818_PDX_PAR1006_gfp_100x15um_set1.mha', 'S:/Tristan/Mask/20200818_PDX_PAR1006_gfp_100x15um_set1_mask.mha', 0.1, 1.3, 2, 1, 0, 0)
    Run('S:/Tristan/Upsampled/20200902_PDX_PAR1059_gfp_100x15um_set1.mha', 'S:/Tristan/Mask/20200902_PDX_PAR1059_gfp_100x15um_set1_mask.mha', 0.04, 1.5, 2, 0, 1, 0)
    Run('S:/Tristan/Upsampled/20210105_PDX_STG316_GFP_rep1_100x15um.mha', 'S:/Tristan/Mask/20210105_PDX_STG316_GFP_rep1_100x15um_mask.mha', 0.04, 0.15, 2, 0, 1, 0)
    Run('S:/Tristan/Upsampled/20201207_PDX_PAR1059x2GFP_100x15um.mha', 'S:/Tristan/Mask/20201207_PDX_PAR1059x2GFP_100x15um_mask.mha', 0.04, 1.5, 2, 0, 1, 0)
    Run('S:/Tristan/Upsampled/20200908_PDX_PAR1022_gfp_100x15um_set1.mha', 'S:/Tristan/Mask/20200908_PDX_PAR1022_gfp_100x15um_set1_mask.mha', 0.04, 1.5, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20201103_PDX_PAR1022_GFP_Rep2_100x15um.mha', 'S:/Tristan/Mask/20201103_PDX_PAR1022_GFP_Rep2_100x15um_mask.mha', 0.04, 1.5, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20210119_PDX_STG316_GFP_100x15um.mha', 'S:/Tristan/Mask/20210119_PDX_STG316_GFP_100x15um_mask.mha', 0.04, 0.5, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20210121_PDX_STG316_GFP_001389_100x15um.mha', 'S:/Tristan/Mask/20210121_PDX_STG316_GFP_001389_100x15um_mask.mha', 0.04, 0.5, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20210104_PDX_AB580Sc_GFP_100x15um.mha', 'S:/Tristan/Mask/20210104_PDX_AB580Sc_GFP_100x15um_mask.mha', 0, 0.3, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20201216_PDX_STG143_SC_GFP_100x15um.mha', 'S:/Tristan/Mask/20201216_PDX_STG143_SC_GFP_100x15um_mask.mha', 0.05, 1, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20201109_PDX_STG1394_GFP_100x15um.mha', 'S:/Tristan/Mask/20201109_PDX_STG1394_GFP_100x15um_mask.mha', 0.1, 1, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20201110_PDX_HC1010_GFP_100x15um.mha', 'S:/Tristan/Mask/20201110_PDX_HC1010_GFP_100x15um_mask.mha', 0, 1, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20210211_PDX_PAR1006_GFP_100x15um.mha', 'S:/Tristan/Mask/20210211_PDX_PAR1006_GFP_100x15um.mha', 0.1, 1, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20210127_PDX_AB630_GFP_100x15um.mha', 'S:/Tristan/Mask/20210127_PDX_AB630_GFP_100x15um_mask.mha', 0.1, 1, 2, 1, 1, 0)
    Run('S:/Tristan/Upsampled/20210126_AB764_017377_100x15um_gfp.mha', 'S:/Tristan/Mask/20210126_AB764_017377_100x15um_gfp_mask.mha', 0.1, 1, 2, 1, 1, 0)


    #problem Run('S:/Tristan/Upsampled/20210111_PDX_STG143_2HC_GFP_100x15um.mha', 'S:/Tristan/Mask/20210111_PDX_STG143_2HC_GFP_100x15um_mask.mha', 0.01, 0.05, 2, 1, 1, 0)



import argparse
parser = argparse.ArgumentParser(description='Segment Stroma.')
parser.add_argument('file_name')
parser.add_argument('output_file_name')
parser.add_argument('-threshold', type=float)
parser.add_argument('-alpha', type=float)
parser.add_argument('-alpha_1d', type=float)
parser.add_argument('-min_size_tissue', type=float)
parser.add_argument('-min_size_holes', type=float)

args = parser.parse_args()

Run(args.file_name, args.output_file_name, args.threshold, args.alpha, args.alpha_1d, args.min_size_tissue, args.min_size_holes)


