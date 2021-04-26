

import cv2
from scipy import ndimage
import numpy as np
import xarray as xr
import sys
import argparse
import SimpleITK as sitk


def Load(file_name, channel_to_load,  resolution):

    global align_x
    global align_y
    global slice_names
    global ds1

    print(file_name)

    try:
        ds1 = xr.open_zarr(file_name, consolidated=True)
        print("consolidated")
    except Exception:
        print("none-consolidated")
        ds1 = xr.open_zarr(file_name)

    slice_names = ds1.attrs['cube_reg']['slice']

    bscale = ds1.attrs['bscale']
    bzero = ds1.attrs['bzero']

    align_x = ds1.attrs['cube_reg']['abs_dx']
    align_y = ds1.attrs['cube_reg']['abs_dy']


    print("loading at resolution {}".format(resolution))
    index = 0
    if resolution == 2:
        index = 1
    elif resolution == 4:
        index = 2
    elif resolution == 8:
        index = 3
    elif resolution == 16:
        index = 4
    elif resolution == 32:
        index = 5

    print("group: " + ds1.attrs["multiscale"]['datasets'][index]['path'])
    gr = ds1.attrs["multiscale"]['datasets'][index]['path']
    ds = xr.open_zarr(file_name, group=gr)

    volume = (ds.sel(channel=channel_to_load, type='mosaic',
              z=0).to_array().data * bscale + bzero)
    # volume_2 = (ds.sel(channel=2, type='mosaic', z=0).to_array().data * bscale + bzero)
    # volume_3 = (ds.sel(channel=3, type='mosaic', z=0).to_array().data * bscale + bzero)
    # volume_4 = (ds.sel(channel=4, type='mosaic', z=0).to_array().data * bscale + bzero)

    return volume


def Align(volume, resolution, output_resolution):

    global align_x
    global align_y
    global slice_names
    global ds1

    spacing = (ds1[slice_names[0]].attrs['scale'])
    size_multiplier = (resolution*0.1*spacing[0])/output_resolution

    size = (volume.shape[0], int(size_multiplier *
            volume.shape[1]), int(size_multiplier*volume.shape[2]))
    aligned = np.zeros(size, dtype=float)

    size2D = (
        int(size_multiplier*volume.shape[2]), int(size_multiplier*volume.shape[1]))

    z_size = volume.shape[0]
    for z in range(0, z_size):
        print('{}/{}'.format(z, z_size))
        fixed = sitk.GetImageFromArray(volume[z, :, :])
        fixed.SetOrigin((0, 0))

        slice_name = slice_names[z]
        spacing = (ds1[slice_name].attrs['scale'])

        fixed.SetSpacing([resolution*0.1*spacing[1],
                         resolution*0.1*spacing[0]])

        print('{},{}'.format(align_y[z], align_x[z]))

        transform = sitk.Euler2DTransform()

        alignY = 0
        if not np.isnan(align_y[z]):
            alignY = -align_y[z]*0.1*spacing[1]

        alignX = 0
        if not np.isnan(align_x[z]):
            alignX = -align_x[z]*0.1*spacing[0]

        transform.SetTranslation([alignY, alignX])

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(size2D)
        resampler.SetOutputSpacing([output_resolution, output_resolution])
        resampler.SetOutputOrigin((0, 0))
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-10)
        resampler.SetTransform(transform)
        out = resampler.Execute(fixed)

        np_out = sitk.GetArrayFromImage(out)
        aligned[z, :, :] = np_out

    return aligned


def GetThreshold(volume):

    tissue = volume > -10
    objs = ndimage.find_objects(tissue)
    maxX = int(objs[0][0].stop)
    minX = int(objs[0][0].start)
    maxY = int(objs[0][1].stop)
    minY = int(objs[0][1].start)
    maxZ = int(objs[0][2].stop)
    minZ = int(objs[0][2].start)

    print("getting threshold in range {}-{},{}-{},{}-{}".format(minX,
          maxX, minY, maxY, minZ, maxZ))

    # median = npmedian(volume[minX:maxX, minY:maxY, minZ:maxZ])
    # threshold = np.bincount(volume[minX:maxX, minY:maxY, minZ:maxZ].ravel()).argmax()

    hist, bin_edges = np.histogram(
        volume[minX:maxX, minY:maxY, minZ:maxZ].ravel(), bins=100)

    threshold1 = bin_edges[np.argmax(hist)]
    threshold2 = bin_edges[np.argmax(hist)+1]

    print("threshold: {}".format(threshold1))
    print("threshold: {}".format(threshold2))

    return threshold2

def GetThreshold2(section):

    tissue = section > -10
    objs = ndimage.find_objects(tissue)
    maxX = int(objs[0][0].stop)
    minX = int(objs[0][0].start)
    maxY = int(objs[0][1].stop)
    minY = int(objs[0][1].start)

    print("getting threshold in range {}-{},{}-{}".format(minX,
          maxX, minY, maxY))

    hist, bin_edges = np.histogram(
        section[minX:maxX, minY:maxY].ravel(), bins=1000)

    threshold1 = bin_edges[np.argmax(hist)]
    index = np.argmax(hist)+10
    if index > 1000:
        index = 1000
    threshold2 = bin_edges[index]

    print("threshold: {}".format(threshold1))
    print("threshold: {}".format(threshold2))

    return threshold2


def Remove_Regions(volume, threshold_, max_size):

    for z in range(0, volume.shape[0]):
        print('{}/{}'.format(z+1, volume.shape[0]))

        threshold = GetThreshold2(volume[z, :, :])
        if threshold > threshold_:
            threshold_ = threshold
        
        thresholded_z = volume[z, :, :] > threshold
        thresholded_z = ndimage.binary_fill_holes(thresholded_z)
        thresholded_z = thresholded_z.astype(np.uint8)

        volume_z = volume[z, :, :]

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            thresholded_z, connectivity=4)
        sizes = stats[:, -1]

        # if use_size:
        #    max_size = float(self.maxSizeN.text())
        # else:
        sizes_sorted = np.sort(sizes, axis=0)
        keep_n = 1
        max_size = sizes_sorted[len(sizes_sorted)-1-keep_n]

        for i in range(1, nb_components):
            if sizes[i] < max_size:
                volume_z[output == i] = threshold

        volume[z, :, :] = volume_z

    return volume, threshold_


def Crop(volume_1, volume_2, volume_3, volume_4, threshold_1, threshold_2, threshold_3, threshold_4):

    maxX_final = 0
    minX_final = sys.maxsize
    maxY_final = 0
    minY_final = sys.maxsize
    maxZ_final = 0
    minZ_final = sys.maxsize

    if volume_1 is not None:

        tissue = volume_1 > threshold_1
        objs = ndimage.find_objects(tissue)
        # maxX = int(objs[0][0].stop)
        # minX = int(objs[0][0].start)
        maxY = int(objs[0][1].stop)
        minY = int(objs[0][1].start)
        maxZ = int(objs[0][2].stop)
        minZ = int(objs[0][2].start)

        # maxX_final = max(maxX_final, maxX)
        # minX_final = min(minX_final, minX)
        maxY_final = max(maxY_final, maxY)
        minY_final = min(minY_final, minY)
        maxZ_final = max(maxZ_final, maxZ)
        minZ_final = min(minZ_final, minZ)

        minX_final = 0
        maxX_final = volume_1.shape[0]

    if volume_2 is not None:

        tissue = volume_2 > threshold_2
        objs = ndimage.find_objects(tissue)
        # maxX = int(objs[0][0].stop)
        # minX = int(objs[0][0].start)
        maxY = int(objs[0][1].stop)
        minY = int(objs[0][1].start)
        maxZ = int(objs[0][2].stop)
        minZ = int(objs[0][2].start)

        # maxX_final = max(maxX_final, maxX)
        # minX_final = min(minX_final, minX)
        maxY_final = max(maxY_final, maxY)
        minY_final = min(minY_final, minY)
        maxZ_final = max(maxZ_final, maxZ)
        minZ_final = min(minZ_final, minZ)

        minX_final = 0
        maxX_final = volume_2.shape[0]

    if volume_3 is not None:

        tissue = volume_3 > threshold_3
        objs = ndimage.find_objects(tissue)
        # maxX = int(objs[0][0].stop)
        # minX = int(objs[0][0].start)
        maxY = int(objs[0][1].stop)
        minY = int(objs[0][1].start)
        maxZ = int(objs[0][2].stop)
        minZ = int(objs[0][2].start)

        # maxX_final = max(maxX_final, maxX)
        # minX_final = min(minX_final, minX)
        maxY_final = max(maxY_final, maxY)
        minY_final = min(minY_final, minY)
        maxZ_final = max(maxZ_final, maxZ)
        minZ_final = min(minZ_final, minZ)

        minX_final = 0
        maxX_final = volume_3.shape[0]

    if volume_4 is not None:

        tissue = volume_4 > threshold_4
        objs = ndimage.find_objects(tissue)
        # maxX = int(objs[0][0].stop)
        # minX = int(objs[0][0].start)
        maxY = int(objs[0][1].stop)
        minY = int(objs[0][1].start)
        maxZ = int(objs[0][2].stop)
        minZ = int(objs[0][2].start)

        # maxX_final = max(maxX_final, maxX)
        # minX_final = min(minX_final, minX)
        maxY_final = max(maxY_final, maxY)
        minY_final = min(minY_final, minY)
        maxZ_final = max(maxZ_final, maxZ)
        minZ_final = min(minZ_final, minZ)

        minX_final = 0
        maxX_final = volume_4.shape[0]

    print("cropping to {}-{},{}-{},{}-{}".format(minX_final,
          maxX_final, minY_final, maxY_final, minZ_final, maxZ_final))

    if volume_1 is not None:
        volume_1 = volume_1[minX_final:maxX_final,
                            minY_final:maxY_final, minZ_final:maxZ_final]
    if volume_2 is not None:
        volume_2 = volume_2[minX_final:maxX_final,
                            minY_final:maxY_final, minZ_final:maxZ_final]
    if volume_3 is not None:
        volume_3 = volume_3[minX_final:maxX_final,
                            minY_final:maxY_final, minZ_final:maxZ_final]
    if volume_4 is not None:
        volume_4 = volume_4[minX_final:maxX_final,
                            minY_final:maxY_final, minZ_final:maxZ_final]

    return volume_1, volume_2, volume_3, volume_4



def Upsample(volume, output_resolution):
    
    upsampled = np.zeros(((2*volume.shape[0])-1,volume.shape[1],volume.shape[2]), dtype=float)
    
    z_size = volume.shape[0]
    for z in range(0, z_size-1):
        print("-------")
        print('{}/{}'.format(z,z_size))
        fixed = sitk.GetImageFromArray(volume[z,:,:])
        moving = sitk.GetImageFromArray(volume[z+1,:,:])

        fixed.SetOrigin((0, 0))
        fixed.SetSpacing([output_resolution,output_resolution])
        moving.SetOrigin((0, 0))
        moving.SetSpacing([output_resolution,output_resolution])

        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(1024)
        matcher.SetNumberOfMatchPoints(7)
        matcher.ThresholdAtMeanIntensityOff()
        moving = matcher.Execute(moving, fixed)

        # The fast symmetric forces Demons Registration Filter
        # Note there is a whole family of Demons Registration algorithms included in
        # SimpleITK
        demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        demons.SetNumberOfIterations(10)
        # Standard deviation for Gaussian smoothing of displacement field
        demons.SetStandardDeviations(1.0)
        #print("running deformable registration")
        displacementField = demons.Execute(fixed, moving)
        #print("Number Of Iterations: {0}".format(demons.GetElapsedIterations()))
        print(" RMS: {0}".format(demons.GetRMSChange()))
        #print("inverting displacement field")
        displacementField_inv = sitk.InvertDisplacementField(displacementField, 10)
        
        np_displacementField = sitk.GetArrayFromImage(displacementField)
        np_displacementField *= 0.5
        displacementField = sitk.GetImageFromArray(np_displacementField, isVector=True)

        np_displacementField_inv = sitk.GetArrayFromImage(displacementField_inv)
        np_displacementField_inv *= 0.5
        displacementField_inv = sitk.GetImageFromArray(np_displacementField_inv, isVector=True)
        
        print("inverting displacement fields")
        displacementField2 = sitk.InvertDisplacementField(displacementField, 10)
        displacementField_inv2 = sitk.InvertDisplacementField(displacementField_inv, 10)

        print("get interpolated slice")
        outTx = sitk.DisplacementFieldTransform(displacementField2)
        outTx_inv = sitk.DisplacementFieldTransform(displacementField_inv2)

        #sitk.WriteTransform(outTx, "H:/GitRepositories/ITK_test/out3.mhd")

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(20)
        resampler.SetTransform(outTx_inv)

        out = resampler.Execute(moving)
        
        resampler_inv = sitk.ResampleImageFilter()
        resampler_inv.SetReferenceImage(moving)
        resampler_inv.SetInterpolator(sitk.sitkLinear)
        resampler_inv.SetDefaultPixelValue(20)
        resampler_inv.SetTransform(outTx)

        out_inv = resampler_inv.Execute(fixed)

        np_out = sitk.GetArrayFromImage(out)
        np_out_inv = sitk.GetArrayFromImage(out_inv)

        upsampled[(2*z)+1,:,:] = (np_out + np_out_inv) / 2.0

        upsampled[2*z,:,:] = volume[z,:,:]
        
    upsampled[(2*z_size)-2,:,:] = volume[z_size-1,:,:]
    
    return upsampled


def WriteImage(file_name, volume, spacing):
    volume_itk = sitk.GetImageFromArray(volume)
    volume_itk.SetSpacing(spacing)

    # print(volume_itk.GetSize())
    # print(volume_itk.GetOrigin())
    # print(volume_itk.GetSpacing())
    # print(volume_itk.GetDirection())
    # print(volume_itk.GetNumberOfComponentsPerPixel())
    # print(volume_itk.GetWidth())
    # print(volume_itk.GetHeight())
    # print(volume_itk.GetDepth())
    # print(volume_itk.GetDimension())
    # print(volume_itk.GetPixelIDValue())
    # print(volume_itk.GetPixelIDTypeAsString())
    # print(volume_itk.GetNumberOfComponentsPerPixel())

    caster = sitk.CastImageFilter()
    # caster.SetOutputPixelType(sitk.sitkFloat32)
    caster.SetOutputPixelType(sitk.sitkVectorFloat32)

    volume_itk = caster.Execute(volume_itk)

    sitk.WriteImage(volume_itk, file_name)


def Run(file_name, output_file_name, exclude_C1, exclude_C2, exclude_C3, exclude_C4, resolution=16, output_resolution=7.5, threshold=None):

    volume_1 = None
    volume_2 = None
    volume_3 = None
    volume_4 = None

    threshold_1 = threshold
    threshold_2 = threshold
    threshold_3 = threshold
    threshold_4 = threshold

    spacing = (output_resolution, output_resolution, 15)
    min_region_size = 200000 / (output_resolution*output_resolution)

    if not exclude_C1:
        volume_1 = Load(file_name, 1, resolution)
        volume_1 = Align(volume_1, resolution, output_resolution)
        volume_1 = volume_1.astype(float)
        #if threshold_1 is None:
        #    threshold_1 = GetThreshold(volume_1)
        threshold_1 = -20
        (volume_1, threshold_1) = Remove_Regions(volume_1, threshold_1, min_region_size)

    if not exclude_C2:
        volume_2 = Load(file_name, 2, resolution)
        volume_2 = Align(volume_2, resolution, output_resolution)
        volume_2 = volume_2.astype(float)
        #if threshold_2 is None:
        #    threshold_2 = GetThreshold(volume_2)
        threshold_2 = -20
        (volume_2, threshold_2) = Remove_Regions(volume_2, threshold_2, min_region_size)
        print("threshold {}".format(threshold_2))

    if not exclude_C3:
        volume_3 = Load(file_name, 3, resolution)
        volume_3 = Align(volume_3, resolution, output_resolution)
        volume_3 = volume_3.astype(float)
        #if threshold_3 is None:
        #    threshold_3 = GetThreshold(volume_3)
        threshold_3 = -20
        (volume_3, threshold_3) = Remove_Regions(volume_3, threshold_3, min_region_size)

    if not exclude_C4:
        volume_4 = Load(file_name, 4, resolution)
        volume_4 = Align(volume_4, resolution, output_resolution)
        volume_4 = volume_4.astype(float)
        #if threshold_4 is None:
        #    threshold_4 = GetThreshold(volume_4)
        threshold_4 = -20
        (volume_4, threshold_4) = Remove_Regions(volume_4, threshold_4, min_region_size)


    (volume_1, volume_2, volume_3, volume_4) = Crop(volume_1, volume_2,
                                                    volume_3, volume_4, threshold_1, threshold_2, threshold_3, threshold_4)


    #volume_1 = Upsample(volume_1, output_resolution)
    #volume_2 = Upsample(volume_2, output_resolution)
    #volume_3 = Upsample(volume_3, output_resolution)
    #volume_4 = Upsample(volume_4, output_resolution)

    if not exclude_C1:
        volume = volume_1
        try:
            volume = np.stack((volume, volume_2), axis=3)
        except:
            pass
        try:
            volume = np.stack((volume, volume_3), axis=3)
        except:
            pass
        try:
            volume = np.stack((volume, volume_4), axis=3)
        except:
            pass

    elif not exclude_C2:
        volume = volume_2
        try:
            volume = np.stack((volume, volume_3), axis=3)
        except:
            pass
        try:
            volume = np.stack((volume, volume_4), axis=3)
        except:
            pass

    elif not exclude_C3:
        volume = volume_3
        try:
            volume = np.stack((volume, volume_4), axis=3)
        except:
            pass
    else:
        volume = volume_3


    #volume = np.stack((volume_2, volume_3), axis=3)
    #print(volume.shape)

    WriteImage(output_file_name, volume, spacing)
    #io.imsave('H:/GitRepositories/STPT/test.tif',volume)


#Run('N:/stpt/20190911_PDX_STG316_gfp_100x15um/mos.zarr', 'S:/Tristan/20190911_PDX_STG316_gfp_100x15um.tif', resolution = 16, output_resolution = 32)
#Run('N:/stpt/20210223_MPR_infl_lung_NSG_GFP_Tdtom_Day19_200x15um/mos.zarr', 'S:/Tristan/20210223_MPR_infl_lung_NSG_GFP_Tdtom_Day19_200x15um.tif', resolution=16, output_resolution=15)


parser = argparse.ArgumentParser(description='Get STPT image.')
parser.add_argument('file_name')
parser.add_argument('output_file_name')
parser.add_argument('-resolution', type=int, default=32)
parser.add_argument('-output_resolution', type = int, default=15)
parser.add_argument('-threshold', type = float, default=None)
parser.add_argument("-exclude_C1",action="store_true",help="exclude channel 1 from the output image")
parser.add_argument("-exclude_C2",action="store_true",help="exclude channel 2 from the output image")
parser.add_argument("-exclude_C3",action="store_true",help="exclude channel 3 from the output image")
parser.add_argument("-exclude_C4",action="store_true",help="exclude channel 4 from the output image")
args = parser.parse_args()
Run(args.file_name, args.output_file_name, args.exclude_C1, args.exclude_C2, args.exclude_C3, args.exclude_C4, args.resolution, args.output_resolution, args.threshold)


#python ReformatSTPT.py "N:/stpt/20190911_PDX_STG316_gfp_100x15um/mos.zarr" "S:/Tristan/20190911_PDX_STG316_gfp_100x15um.tif" -resolution=32 -exclude_C1 -exclude_C4

#python ReformatSTPT.py "N:/stpt/20210223_MPR_infl_lung_NSG_GFP_Tdtom_Day19_200x15um/mos.zarr" "S:/Tristan/20190911_PDX_STG316_gfp_100x15um.tif" -resolution=32 -exclude_C1 -exclude_C4