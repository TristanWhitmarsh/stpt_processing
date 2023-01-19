
from skimage.transform import pyramid_gaussian
from skimage import data
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square, remove_small_objects, binary_opening, remove_small_holes, binary_erosion, binary_dilation
from skimage import measure
from skimage import data
from skimage.morphology import skeletonize, disk, ball
from skimage.filters import threshold_local, threshold_sauvola, threshold_niblack, threshold_otsu, rank



#import cv2
#from sklearn.preprocessing import normalize
from scipy import ndimage
from scipy import stats
import os
#import itk
import dask
#import napari
import numpy as np
import xarray as xr
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QHBoxLayout, QGroupBox, QVBoxLayout, QGridLayout, QSpacerItem, QLineEdit, QLabel, QComboBox

from PIL import Image
import trimesh

import statistics
from scipy import ndimage
#import cv2
import math
#import openmesh as om

import SimpleITK as sitk

import matplotlib.pyplot as plt
import gc
from PIL import Image
import time

def GetStromaOld(volume, tissue, thresh = 0.3, block_size = 101):

    offset = 0
    #minimum_size = 1000

    stroma = np.zeros(volume.shape, dtype=int)
    
    #adaptive = np.zeros(volume.shape, dtype=float)
    radius = 101
    selem = disk(radius)
    
    
    #volume = volume.astype(np.float32)
    #volume = volume / 50.0
    
    for i in range(0, volume.shape[0]):
        volume_itk = sitk.GetImageFromArray(volume[i,:,:])
        #tissue_itk = sitk.GetImageFromArray(tissue[i,:,:])
        #labeled_itk = sitk.OtsuThreshold(volume_itk, tissue_itk, 0, 1, 128, True, 1)
        #labeled_itk = sitk.OtsuMultipleThresholdsImageFilter(volume_itk)
         
        #labeled = sitk.GetArrayFromImage(labeled_itk)
        #stroma[i,:,:] = tissue[i,:,:]
        #stroma[i,:,:] = labeled
        adaptive_thresh = threshold_local(volume[i,:,:], block_size, offset=offset)
        #adaptive_thresh = threshold_sauvola(volume[i,:,:], window_size=31, k=0)
        #adaptive_thresh = rank.otsu(volume[i,:,:], selem)
        #adaptive[i,:,:] = adaptive_thresh
        stroma[i,:,:] = volume[i,:,:] > adaptive_thresh
        #stroma[i,:,:] += volume[i,:,:] > 2
        
        #stroma[i,:,:] = threshold_niblack(volume[i,:,:], window_size=31, k=0)

    #adaptive_itk = sitk.GetImageFromArray(adaptive)
    #sitk.WriteImage(adaptive_itk, 'S:/Tristan/adaptive_itk.mha')
    
    #stroma = stroma > 0
    #stroma = stroma.astype(np.uint8)

    #stroma_itk = sitk.GetImageFromArray(stroma)
    #sitk.WriteImage(stroma_itk, 'S:/Tristan/stroma_itk.mha')

    

    #volume_itk = sitk.GetImageFromArray(volume)
    #tissue = tissue.astype(np.uint8)
    #tissue_itk = sitk.GetImageFromArray(tissue)
    #labeled_itk = sitk.OtsuThreshold(volume_itk)
    #sitk.WriteImage(labeled_itk, 'S:/Tristan/labeled_itk.mha')

    #stroma = stroma > 0
    #stroma = remove_small_objects(stroma, min_size=minimum_size, in_place=False)
    #stroma = stroma.astype(np.uint8)
    #stroma_itk = sitk.GetImageFromArray(stroma)
    #sitk.WriteImage(stroma_itk, 'S:/Tristan/stroma_filtered_itk.mha')

    #smooth_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    #smooth_filter.SetSigma(5)
    #stroma_itk = smooth_filter.Execute(stroma_itk)
    #stroma = sitk.GetArrayFromImage(stroma_itk)
    #stroma = stroma > 0.5
    #stroma = np.asarray(stroma).astype(np.uint8)
    #stroma_itk = sitk.GetImageFromArray(stroma)
    #sitk.WriteImage(stroma_itk, 'S:/Tristan/STPT/smoothed_itk.mha')
    

    return stroma

def GetStroma(volume, thresh = 0.3, alpha = 1, alpha_1d = 1, adaptive = False):

    if adaptive:
        print("running adaptive thresholding")
        offset = 0
        stroma = np.zeros(volume.shape, dtype=int)
        radius = 101
        selem = disk(radius)
        
        for i in range(0, volume.shape[0]):
            volume_itk = sitk.GetImageFromArray(volume[i,:,:])
            adaptive_thresh = threshold_local(volume[i,:,:], radius, offset=offset)
            stroma[i,:,:] = volume[i,:,:] > adaptive_thresh
    else:
        print("running simple thresholding")
        stroma = volume > thresh



    #stroma = np.asarray(stroma).astype(np.uint8)
    stroma = stroma.astype(np.float)
    if alpha_1d > 0:
        stroma = ndimage.gaussian_filter1d(stroma, alpha_1d, 0)

    if alpha > 0:
        stroma = ndimage.gaussian_filter(stroma, alpha)
    #stroma_itk = sitk.GetImageFromArray(stroma)
    #smooth_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    #smooth_filter.SetSigma(2)
    #stroma_itk = smooth_filter.Execute(stroma_itk)
    #stroma = sitk.GetArrayFromImage(stroma_itk)
    stroma = stroma > 0.5
    #stroma = np.asarray(stroma).astype(np.uint8)

    #stroma_itk = sitk.GetImageFromArray(stroma)
    #median_filter = sitk.BinaryMedianImageFilter()
    #median_filter.SetRadius(1)
    #stroma_itk = median_filter.Execute(stroma_itk)
    #stroma = sitk.GetArrayFromImage(stroma_itk)

    #stroma = stroma > 0
    stroma = remove_small_objects(stroma, 100)
    stroma = remove_small_holes(stroma, 100)

    
    #stroma = np.asarray(stroma).astype(np.uint8)
    


    #stroma = np.asarray(tissue).astype(np.uint8)
    #stroma_itk = sitk.GetImageFromArray(stroma)
    #sitk.WriteImage(tissue_itk, 'S:/Tristan/tissue_itk.mha')

    #smooth_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    #smooth_filter.SetSigma(5)
    #tissue_itk = smooth_filter.Execute(tissue_itk)
    #tissue_itk = sitk.Threshold(tissue_itk, 0.5, 1)
    #sitk.WriteImage(tissue_itk, 'S:/Tristan/smoothed_itk.mha')

    #smooth_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    #smooth_filter.SetSigma(5)
    #tissue_itk = smooth_filter.Execute(tissue_itk)
    #tissue = sitk.GetArrayFromImage(tissue_itk)
    #tissue = tissue > 0.5
    #tissue = np.asarray(tissue).astype(np.uint8)
    #tissue_itk = sitk.GetImageFromArray(tissue)
    #sitk.WriteImage(tissue_itk, 'S:/Tristan/smoothed_itk.mha')

    #connection_filter = sitk.ConnectedComponentImageFilter()
    #tissue_itk = connection_filter.Execute(tissue_itk)
    #relabel_filter = sitk.RelabelComponentImageFilter()
    #tissue_itk = relabel_filter.Execute(tissue_itk)
    #tissue_itk = sitk.Threshold(tissue_itk, 0.5, 1.5)
    #sitk.WriteImage(tissue_itk, 'S:/Tristan/largest_itk.mha')
    
    #fill_filter = sitk.BinaryFillholeImageFilter()
    #fill_filter.SetForegroundValue(1)
    #tissue_itk = fill_filter.Execute(tissue_itk)
    #sitk.WriteImage(tissue_itk, 'S:/Tristan/filled_itk.mha')
    

    #smooth_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    #smooth_filter.SetSigma(10)
    #tissue_itk = smooth_filter.Execute(tissue_itk)
    #tissue = sitk.GetArrayFromImage(tissue_itk)
    #tissue = tissue > 0.5
    #tissue = np.asarray(tissue).astype(np.uint8)
    #tissue_itk = sitk.GetImageFromArray(tissue)
    #sitk.WriteImage(tissue_itk, 'S:/Tristan/smoothed_again_itk.mha')

    #smooth_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    #smooth_filter.SetSigma(5)
    #tissue_itk = smooth_filter.Execute(tissue_itk)
    #tissue_itk = sitk.Threshold(tissue_itk, 0.5, 1)
    #sitk.WriteImage(tissue_itk, 'S:/Tristan/smoothed_again_itk.mha')

    return stroma

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def GetTissue(volume, thresh = 1, alpha = 2):
    
    tissue = volume > thresh
    tissue = tissue.astype(np.float)

    if alpha > 0:
        tissue = ndimage.gaussian_filter(tissue, alpha)
    tissue = tissue > 0.5
    tissue = getLargestCC(tissue)

    #struct = ball(10)
    #tissue = ndimage.binary_erosion(tissue, structure=struct, iterations=1)
    
    
    #tissue = np.asarray(tissue).astype(np.uint8)
    #tissue_itk = sitk.GetImageFromArray(tissue)
    #sitk.WriteImage(tissue_itk, 'S:/Tristan/tissue_itk.mha')

    #smooth_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    #smooth_filter.SetSigma(5)
    #tissue_itk = smooth_filter.Execute(tissue_itk)
    #tissue_itk = sitk.Threshold(tissue_itk, 0.5, 1)
    #sitk.WriteImage(tissue_itk, 'S:/Tristan/smoothed_itk.mha')

    #smooth_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    #smooth_filter.SetSigma(2)
    #tissue_itk = smooth_filter.Execute(tissue_itk)
    #tissue = sitk.GetArrayFromImage(tissue_itk)
    #tissue = tissue > 0.5
    #tissue = np.asarray(tissue).astype(np.uint8)
    #tissue_itk = sitk.GetImageFromArray(tissue)
    #sitk.WriteImage(tissue_itk, 'S:/Tristan/smoothed_itk.mha')

    #connection_filter = sitk.ConnectedComponentImageFilter()
    #tissue_itk = connection_filter.Execute(tissue_itk)
    #relabel_filter = sitk.RelabelComponentImageFilter()
    #tissue_itk = relabel_filter.Execute(tissue_itk)
    #tissue_itk = sitk.Threshold(tissue_itk, 0.5, 1.5)
    #sitk.WriteImage(tissue_itk, 'S:/Tristan/largest_itk.mha')
    
    #fill_filter = sitk.BinaryFillholeImageFilter()
    #fill_filter.SetForegroundValue(1)
    #tissue_itk = fill_filter.Execute(tissue_itk)
    #sitk.WriteImage(tissue_itk, 'S:/Tristan/filled_itk.mha')
    
    #erode_filter = sitk.BinaryErodeImageFilter()
    #erode_filter.SetKernelRadius(20)
    #erode_filter.SetForegroundValue(1)
    #tissue_itk = erode_filter.Execute(tissue_itk)
    #sitk.WriteImage(tissue_itk, 'S:/Tristan/eroded_itk.mha')

    #smooth_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    #smooth_filter.SetSigma(10)
    #tissue_itk = smooth_filter.Execute(tissue_itk)
    #tissue = sitk.GetArrayFromImage(tissue_itk)
    #tissue = tissue > 0.5
    #tissue = np.asarray(tissue).astype(np.uint8)
    #tissue_itk = sitk.GetImageFromArray(tissue)
    #sitk.WriteImage(tissue_itk, 'S:/Tristan/smoothed_again_itk.mha')

    #smooth_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    #smooth_filter.SetSigma(5)
    #tissue_itk = smooth_filter.Execute(tissue_itk)
    #tissue_itk = sitk.Threshold(tissue_itk, 0.5, 1)
    #sitk.WriteImage(tissue_itk, 'S:/Tristan/smoothed_again_itk.mha')

    #tissue = sitk.GetArrayFromImage(tissue_itk)

    return tissue




def Run(file_name, output_file_name, threshold_tissue, threshold_stroma, alpha_tissue=2, alpha_stroma=1, alpha_stroma_1d = 1, adaptive = 0):
    print(threshold_tissue)
    print(threshold_stroma)
    print(alpha_tissue)
    print(alpha_stroma)
    print(alpha_stroma_1d)
    
    volume_itk = sitk.ReadImage(file_name, imageIO="MetaImageIO")
    
    #otsu_filter = sitk.OtsuMultipleThresholdsImageFilter()
    #otsu_filter.SetNumberOfThresholds(3)
    #otsu_itk = otsu_filter.Execute(volume_itk) 
    #sitk.WriteImage(otsu_itk, 'S:/Tristan/otsu_itk.mha')

    volume = sitk.GetArrayFromImage(volume_itk)
    
    tissue = GetTissue(volume, threshold_tissue, alpha_tissue)
    tissue = tissue.astype(np.uint8)

    tissue_eroded = np.pad(tissue, ((1, 1), (1, 1), (1, 0)), 'constant', constant_values=1)
    tissue_eroded = ndimage.binary_fill_holes(tissue_eroded)
    tissue_eroded = tissue_eroded[1:tissue_eroded.shape[0]-1,1:tissue_eroded.shape[1]-1,1:]
    tissue_eroded = tissue_eroded.astype(np.uint8)
    for i in range(0, volume.shape[0]):
        print("{}/{}".format(i,volume.shape[0]))

        tissue_itk = sitk.GetImageFromArray(tissue[i,:,:])
        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelRadius(20)
        dilate_filter.SetForegroundValue(1)
        tissue_itk = dilate_filter.Execute(tissue_itk)

        tissue_itk = sitk.BinaryFillhole(tissue_itk)

        erode_filter = sitk.BinaryErodeImageFilter()
        erode_filter.SetKernelRadius(120)
        erode_filter.SetForegroundValue(1)
        tissue_itk = erode_filter.Execute(tissue_itk)

        tissue_eroded[i,:,:] = sitk.GetArrayFromImage(tissue_itk)
    
    tissue[tissue_eroded == 0] = 0





    #tissue255 = 255 * tissue.astype(np.uint8)
    #tissue255_itk = sitk.GetImageFromArray(tissue255)
    #, sitk.sitkUInt8
    #tissue255_itk.SetSpacing(volume_itk.GetSpacing())
    #sitk.WriteImage(tissue_itk, 'S:/Tristan/test_tissue.mha')

    #labeled_itk = sitk.OtsuThreshold(volume_itk)
    #sitk.WriteImage(labeled_itk, 'S:/Tristan/test_label.mha')
    if adaptive == 0:
        stroma = GetStroma(volume, threshold_stroma, alpha_stroma, alpha_stroma_1d, False)
    else:
        stroma = GetStroma(volume, threshold_stroma, alpha_stroma, alpha_stroma_1d, True)

    #stroma = GetStromaOld(volume, tissue, thresh = 0.3, block_size = 101)
    stroma = stroma.astype(np.uint8)
    stroma = tissue * stroma


    #pos = int(tissue.shape[0]/2)
    #im = Image.fromarray(tissue[pos,:,:]*50)
    #im.save("S:/Tristan/tissue.tif")

    #https://stackoverflow.com/questions/33700132/3d-image-erosion-python
    print("done segmenting")
    #tic = time.clock()
    
    tissue_eroded = np.pad(tissue, ((1, 1), (1, 1), (1, 0)), 'constant', constant_values=1)
    tissue_eroded = ndimage.binary_fill_holes(tissue_eroded)
    tissue_eroded = tissue_eroded[1:tissue_eroded.shape[0]-1,1:tissue_eroded.shape[1]-1,1:]
    tissue_eroded = tissue_eroded.astype(np.uint8)
    print("done filling holes") 

    #struct = disk(20)
    #selem = disk(6)

    if True:
        for i in range(0, volume.shape[0]):
            print("{}/{}".format(i,volume.shape[0]))

            tissue_itk = sitk.GetImageFromArray(tissue_eroded[i,:,:])
            dilate_filter = sitk.BinaryDilateImageFilter()
            dilate_filter.SetKernelRadius(20)
            dilate_filter.SetForegroundValue(1)
            tissue_itk = dilate_filter.Execute(tissue_itk)

            tissue_itk = sitk.BinaryFillhole(tissue_itk)

            erode_filter = sitk.BinaryErodeImageFilter()
            erode_filter.SetKernelRadius(220)
            erode_filter.SetForegroundValue(1)
            tissue_itk = erode_filter.Execute(tissue_itk)

            tissue_eroded[i,:,:] = sitk.GetArrayFromImage(tissue_itk)
    else:
        
        struct = ball(10)
        tissue_eroded = ndimage.binary_dilation(tissue_eroded, structure=struct, iterations=2)
        tissue_eroded = ndimage.binary_erosion(tissue_eroded, structure=struct, iterations=22)
        tissue_eroded = tissue_eroded.astype(np.uint8)
        print("done dilating")

    #im = Image.fromarray(tissue_eroded[100,:,:]*50)
    #im.save("S:/Tristan/tissue_dilated.tif")
 
    #return
    #struct = ndimage.generate_binary_structure(3, 20)

    #tissue = np.pad(tissue, ((1, 1), (1, 1), (1, 0)), 'constant', constant_values=1)
    #tissue = ndimage.binary_fill_holes(tissue)
    #tissue = tissue[1:tissue.shape[0]-1,1:tissue.shape[1]-1,1:]

    #tissue_eroded = tissue_eroded.astype(np.uint8)
    #pos = int(tissue_eroded.shape[0]/2)
    #im = Image.fromarray(tissue_eroded[pos,:,:]*50)
    #im.save("S:/Tristan/tissue_eroded.tif")
    #return

    #tissue_eroded2 = ndimage.binary_erosion(tissue_eroded, structure=struct, iterations=40)
    #tissue_eroded2 = tissue_eroded2.astype(np.uint8)
    #im = Image.fromarray(tissue_eroded2[100,:,:]*50)
    #im.save("S:/Tristan/tissue_eroded.tif")

    #toc = time.clock()
    #print(toc - tic)

    #print("done1")
    #tic = time.clock()
    #erodedMask2 = binary_erosion(tissue)
    #erodedMask2 = erodedMask2.astype(np.uint8)
    #im = Image.fromarray(erodedMask2[5,:,:]*50)
    #im.save("S:/Tristan/your_file3.tif")
    #toc = time.clock()
    #print(toc - tic)
    #print("done2")

    #return



    #tissue_itk = sitk.GetImageFromArray(tissue)

    #dilate_filter = sitk.BinaryDilateImageFilter()
    #dilate_filter.SetKernelRadius(10)
    #dilate_filter.SetForegroundValue(1)
    #tissue_itk = dilate_filter.Execute(tissue_itk)

    #print(2)
    #erode_filter = sitk.BinaryErodeImageFilter()
    #erode_filter.SetKernelRadius(10)
    #erode_filter.SetForegroundValue(1)
    #for i in range(0, 12):
    #    print("{}/{}".format(i,12))
    #    tissue_itk = erode_filter.Execute(tissue_itk)
    #tissue_itk = dilate_filter.Execute(tissue_itk)
    
    #tissue_eroded = sitk.GetArrayFromImage(tissue_itk)

    #sitk.WriteImage(tissue_itk, 'S:/Tristan/eroded_itk.mha')
    

    #sitk.WriteImage(stroma_itk, output_file_name)

    #mask = tissue + stroma + (tissue_eroded * 2) 
    

    #print(4)
    mask = tissue + stroma
    mask[(tissue_eroded == 1) & (tissue == 1)] = 3
    mask[(tissue_eroded == 1) & (stroma == 1)] = 4

    
    pos = int(mask.shape[0]/2)
    im = Image.fromarray(mask[pos,:,:]*50)
    im.save(output_file_name[:-4]+"_mask.tif")

    im = Image.fromarray(volume[pos,:,:]*0.1)
    im.save(output_file_name[:-4]+".tif")

    mask = mask.astype(np.uint8)
    mask_itk = sitk.GetImageFromArray(mask)
    mask_itk.SetSpacing(volume_itk.GetSpacing())
    sitk.WriteImage(mask_itk, output_file_name)

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


SegmentStroma()
if False:
    import argparse
    parser = argparse.ArgumentParser(description='Segment Stroma.')
    parser.add_argument('file_name')
    parser.add_argument('output_file_name')
    parser.add_argument('threshold_tissue', type=float)
    parser.add_argument('threshold_stroma', type=float)
    parser.add_argument('alpha_tissue', type=float)
    parser.add_argument('alpha_stroma', type=float)
    parser.add_argument('alpha_stroma_1d', type=float)
    parser.add_argument('adaptive', type=int)
    args = parser.parse_args()

    Run(args.file_name, args.output_file_name, args.threshold_tissue, args.threshold_stroma, args.alpha_tissue, args.alpha_stroma, args.alpha_stroma_1d, args.adaptive)


