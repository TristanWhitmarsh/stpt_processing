
from skimage.transform import pyramid_gaussian
from skimage.segmentation import clear_border
from skimage.morphology import closing, square, remove_small_objects, binary_opening, remove_small_holes, skeletonize_3d, skeletonize, disk, thin, medial_axis
from skimage.filters import threshold_local, threshold_sauvola, threshold_niblack, threshold_otsu, rank
from skimage import data, measure


from scipy import ndimage
from scipy import stats
from PIL import Image
import os
#import napari
import numpy as np
import sys
import trimesh
import statistics
import math
import SimpleITK as sitk
import gc
import porespy as ps
import matplotlib.pyplot as plt



def GetFractalDimension(binary_mask):

    binary_mask_itk = sitk.GetImageFromArray(binary_mask.astype(float))

    counts = []
    sizes = []

    for a in range(1, 6):
        b = pow(2,a)

        binary_mask_itk.SetOrigin((0, 0, 0 ))
        binary_mask_itk.SetSpacing([1,1,1])
        
        size2D = (int(math.floor(binary_mask_itk.GetSize()[0]/2)), int(math.floor(binary_mask_itk.GetSize()[1]/2)), int(math.floor(binary_mask_itk.GetSize()[2]/2)))

        resampler = sitk.ResampleImageFilter()        
        resampler.SetSize(size2D)
        resampler.SetOutputSpacing([2,2,2])
        resampler.SetOutputOrigin((0.5, 0.5, 0.5))
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        binary_mask_itk = resampler.Execute(binary_mask_itk)

        out = sitk.GetArrayFromImage(binary_mask_itk)
        count = np.count_nonzero(out) - np.count_nonzero(out == 1)
        all = out.shape[0] * out.shape[1] * out.shape[2]
        
        if count == all or count <= 1:
            break

        #print("{},{}".format(count, b))
        #print("{},{}".format(math.log(count), math.log(1/b)))
    
        counts.append(math.log(count))
        sizes.append(math.log(1/b))
        
        #sitk.WriteImage(binary_mask_itk, 'S:/Tristan/binary_mask_itk.mha')

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(sizes,counts)
        return slope
    except:
        print("An exception occurred") 
        return 0
    
    #print(slope)



def GetNumberOfHandles(stroma):
    stroma = stroma == 1
    stroma = ndimage.binary_fill_holes(stroma)
    stroma = np.pad(stroma, ((1,1),(1,1),(1,1)), 'constant', constant_values=0)

    blobs, number_of_blobs = ndimage.label(stroma)
    
    verts, faces, normals, values = measure.marching_cubes_lewiner(stroma, level=None, spacing=(1.0, 1.0, 1.0), gradient_direction='descent', step_size=2, allow_degenerate=True, use_classic=False)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    normals = mesh.vertex_normals
    values = normals[:,0]
    surface = (verts, faces, values)

    number_of_vertices = verts.shape[0]
    number_of_edges = (faces.shape[0] * 3)/2
    number_of_cells = faces.shape[0]
    number_of_boundaries = 0

    euler_number = number_of_vertices - number_of_edges + number_of_cells + number_of_boundaries
    #print("euler_number: {}".format(euler_number))
		
    number_of_handles = -(((number_of_vertices - number_of_edges + number_of_cells + number_of_boundaries)/2.0)-1) + (number_of_blobs - 1)
    
    return number_of_handles





def Run(file_name):
    print(file_name)
    #global mask
    #global mask_itk

    mask_itk = sitk.ReadImage(file_name, imageIO="MetaImageIO")
    mask = sitk.GetArrayFromImage(mask_itk)

    tissue_count_full = np.count_nonzero(mask == 1) + np.count_nonzero(mask == 3)
    stroma_count_full = np.count_nonzero(mask == 2) + np.count_nonzero(mask == 4)
    if((tissue_count_full + stroma_count_full) != 0):
        stroma_percentage_full = 100*(stroma_count_full / (tissue_count_full + stroma_count_full))
        print("stroma_percentage_full: {}".format(stroma_percentage_full))
    else:
        stroma_percentage_full = 0

    tissue_count_outer = np.count_nonzero(mask == 1)
    stroma_count_outer = np.count_nonzero(mask == 2)
    if((tissue_count_outer + stroma_count_outer) != 0):
        stroma_percentage_outer = 100*(stroma_count_outer / (tissue_count_outer + stroma_count_outer))
        print("stroma_percentage_outer: {}".format(stroma_percentage_outer))
    else:
        stroma_percentage_outer = 0
    
    tissue_count_inner = np.count_nonzero(mask == 3)
    stroma_count_inner = np.count_nonzero(mask == 4)
    if((tissue_count_inner + stroma_count_inner) != 0):
        stroma_percentage_inner = 100*(stroma_count_inner / (tissue_count_inner + stroma_count_inner))
        print("stroma_percentage_inner: {}".format(stroma_percentage_inner))
    else:
        stroma_percentage_inner = 0

    #stroma_itk = sitk.GetImageFromArray(mask_full.astype(int))
    #sitk.WriteImage(stroma_itk, 'S:/Tristan/stroma_itk1.mha')

    stroma_full = mask == 2
    stroma_full[mask == 4] = 1

    if(stroma_count_full != 0):
        number_of_handles_full = GetNumberOfHandles(stroma_full)
        connectivity_density_full = number_of_handles_full / (0.0075*0.0075*0.0075*(tissue_count_full + stroma_count_full))
        print("number_of_handles_full: {}".format(number_of_handles_full))
        print("connectivity_density_full: {}".format(connectivity_density_full))
        fractal_dimension_full = GetFractalDimension(stroma_full)
        print("fractal_dimension_full: {}".format(fractal_dimension_full))
    else:
        number_of_handles_full = 0
        connectivity_density_full = 0
        fractal_dimension_full = 0
 
    stroma_outer = mask == 2
    if(stroma_count_outer != 0):
        number_of_handles_outer = GetNumberOfHandles(stroma_outer)
        connectivity_density_outer = number_of_handles_outer / (0.0075*0.0075*0.0075*(tissue_count_outer + stroma_count_outer))
        print("number_of_handles_outer: {}".format(number_of_handles_outer))
        print("connectivity_density_outer: {}".format(connectivity_density_outer))
        fractal_dimension_outer = GetFractalDimension(stroma_outer)
        print("fractal_dimension_outer: {}".format(fractal_dimension_outer))
    else:
        number_of_handles_outer = 0
        connectivity_density_outer = 0
        fractal_dimension_outer = 0
    
    stroma_inner = mask == 4
    if(stroma_count_inner != 0):
        number_of_handles_inner = GetNumberOfHandles(stroma_inner)
        connectivity_density_inner = number_of_handles_inner / (0.0075*0.0075*0.0075*(tissue_count_inner + stroma_count_inner))
        print("number_of_handles_inner: {}".format(number_of_handles_inner))
        print("connectivity_density_inner: {}".format(connectivity_density_inner))
        fractal_dimension_inner = GetFractalDimension(stroma_inner)
        print("fractal_dimension_inner: {}".format(fractal_dimension_inner))
    else:
        number_of_handles_inner = 0
        connectivity_density_inner = 0
        fractal_dimension_inner = 0

    #https://github.com/InsightSoftwareConsortium/ITKThickness3D
    #https://discourse.itk.org/t/in-python-how-to-convert-between-simpleitk-and-itk-images/1922
    #https://github.com/PMEAL/porespy/blob/dev/examples/filters/local_thickness.ipynb

    #print("generating thickness map")
    thickness_map = ps.filters.local_thickness(stroma_full, mode='dt')
    thickness_map = np.float32(thickness_map)
    thickness_map_itk = sitk.GetImageFromArray(thickness_map)
    sitk.WriteImage(thickness_map_itk, 'S:/Tristan/Thickness/'+os.path.basename(file_name)[:-9]+'.mha')

    if(np.count_nonzero(stroma_full) > 0):
        average_thickness_full = 7.5 * np.sum(thickness_map) / np.count_nonzero(thickness_map)
        print("np.count_nonzero(thickness_map) {}".format(np.count_nonzero(thickness_map)))

        mask_full = stroma_full == 0
        thickness_map_masked = np.ma.MaskedArray(thickness_map, mask_full)
        standard_deviation_thickness_full = 7.5 * thickness_map_masked.std()
        
        mask_full = None
        thickness_map_masked = None
        #print("average_thickness_full {}".format(average_thickness_full))
        #print("mean {}".format(7.5 * np.ma.MaskedArray(thickness_map, mask).mean()))
        #print("standard_deviation {}".format(standard_deviation_thickness_full))
    else:
        average_thickness_full = 0
        standard_deviation_thickness_full = 0

    thickness_map_outer = np.copy(thickness_map)
    thickness_map_outer[mask == 4] = 0

    if(np.count_nonzero(thickness_map_outer) > 0):
        average_thickness_outer = 7.5 * np.sum(thickness_map_outer) / np.count_nonzero(thickness_map_outer)
        
        mask_outer = thickness_map_outer == 0
        thickness_map_outer_masked = np.ma.MaskedArray(thickness_map_outer, mask_outer)
        standard_deviation_thickness_outer = 7.5 * thickness_map_outer_masked.std()

        mask_outer = None
        thickness_map_outer_masked = None
    else:
        average_thickness_outer = 0
        standard_deviation_thickness_outer = 0

    thickness_map_outer = None

    thickness_map_inner = np.copy(thickness_map)
    thickness_map_inner[mask == 2] = 0
    if(np.count_nonzero(thickness_map_inner) > 0):
        average_thickness_inner = 7.5 * np.sum(thickness_map_inner) / np.count_nonzero(thickness_map_inner)
        
        mask_inner = thickness_map_inner == 0
        thickness_map_inner_masked = np.ma.MaskedArray(thickness_map_inner, mask_inner)
        standard_deviation_thickness_inner = 7.5 * thickness_map_inner_masked.std()

        mask_inner = None
        thickness_map_inner_masked = None
    else:
        average_thickness_inner = 0
        standard_deviation_thickness_inner = 0

    thickness_map_inner = None

    print("average_thickness_full: {}".format(average_thickness_full))
    print("average_thickness_outer: {}".format(average_thickness_outer))
    print("average_thickness_inner: {}".format(average_thickness_inner))
    print("standard_deviation_full: {}".format(standard_deviation_thickness_full))
    print("standard_deviation_outer: {}".format(standard_deviation_thickness_outer))
    print("standard_deviation_inner: {}".format(standard_deviation_thickness_inner))
    
    
    file_object = open('S:/Tristan/measurements.csv', 'a')
    file_object.write(os.path.basename(file_name)[:-9]+','+str(stroma_percentage_full)+','+str(stroma_percentage_outer)+','+str(stroma_percentage_inner)+','+
        str(connectivity_density_full)+','+str(connectivity_density_outer)+','+str(connectivity_density_inner)+','+
        str(fractal_dimension_full)+','+str(fractal_dimension_outer)+','+str(fractal_dimension_inner)+','+
        str(average_thickness_full)+','+str(average_thickness_outer)+','+str(average_thickness_inner)+','+
        str(standard_deviation_thickness_full)+','+str(standard_deviation_thickness_outer)+','+str(standard_deviation_thickness_inner)+'\n')
    file_object.close()

    print("done")
    del thickness_map_inner
    del thickness_map_outer
    del thickness_map
    del stroma_full
    del stroma_inner
    del stroma_outer
    del mask
    gc.collect()

def SegmentStroma():
    gc.collect()


    #file_object = open('S:/Tristan/measurements.csv', 'r+')
    #file_object.write('file_name,stroma_ratio_full,stroma_ratio_outer,stroma_ratio_inner,'+
    #    'connectivity_density_full,connectivity_density_outer,connectivity_density_inner,'+
    #    'fractal_dimension_full,fractal_dimension_outer,fractal_dimension_inner,'+
    #    'average_thickness_full,average_thickness_outer,average_thickness_inner,'+
    #    'standard_deviation_thickness_full,standard_deviation_thickness_outer,standard_deviation_thickness_inner\n')
    #file_object.close()


    Run('S:/Tristan/ForPaper/20190911_PDX_STG316_gfp_100x15um_mask.mha')
    
    return
    
    Run('S:/Tristan/UnitTests/50x50_line.mha')
    Run('S:/Tristan/UnitTests/50x50_plane.mha')
    Run('S:/Tristan/UnitTests/50x50_sphere2.mha')
    Run('S:/Tristan/UnitTests/10x10_double_mask.mha')
    Run('S:/Tristan/UnitTests/10x10_double_mask2.mha')
    Run('S:/Tristan/UnitTests/50x50_all_outer.mha')
    Run('S:/Tristan/UnitTests/50x50_strange.mha')
    Run('S:/Tristan/UnitTests/50x50_cube.mha')
    Run('S:/Tristan/UnitTests/50x50_plane_4.mha')
    Run('S:/Tristan/UnitTests/50x50_plane_5.mha')


    Run('S:/Tristan/Mask/20201109_PDX_STG1394_GFP_100x15um_mask.mha')
    Run('S:/Tristan/Mask/20190911_PDX_STG316_gfp_100x15um_mask.mha')
    Run('S:/Tristan/Mask/20190916_PDX_STG316_gfp_100x15um_set2_mask.mha')
    Run('S:/Tristan/Mask/20190919_PDX_STG139645_gfp_100x15um_mask.mha')
    Run('S:/Tristan/Mask/20191021_PDX_STG143_100x15um_mask.mha')
    Run('S:/Tristan/Mask/20191209_PDX_STG143SC_50x15um_mask.mha')
    Run('S:/Tristan/Mask/20191210_PDX_STG143SC_100x15um_set2_mask.mha')
    Run('S:/Tristan/Mask/20200219_PDX_AB559_50X15um_mask.mha')
    Run('S:/Tristan/Mask/20200225_PDX_Ab559_set2_100X15um_mask.mha')
    Run('S:/Tristan/Mask/20200226_PDX_Ab580_HC_gfp_100x15um_mask.mha')
    Run('S:/Tristan/Mask/20200227_PDX_Ab580_HC_gfp_set2_100x15um_mask.mha')
    Run('S:/Tristan/Mask/20200615_PDX_AB580_GFP_100x15um_set1_mask.mha')
    Run('S:/Tristan/Mask/20200629_PDX_PAR1040_GFP_100x15um_set1_mask.mha')
    Run('S:/Tristan/Mask/20200707_PDX_PAR1040_GFP_100x15um_set2_mask.mha')
    Run('S:/Tristan/Mask/20200804_PDX_AB580SC_gfp_100x15um_set1_mask.mha')
    Run('S:/Tristan/Mask/20200817_PDX_AB764_017376_gfp_100x15um_set1_mask.mha')
    Run('S:/Tristan/Mask/20200818_PDX_PAR1006_gfp_100x15um_set1_mask.mha')
    Run('S:/Tristan/Mask/20200902_PDX_PAR1059_gfp_100x15um_set1_mask.mha')
    Run('S:/Tristan/Mask/20210105_PDX_STG316_GFP_rep1_100x15um_mask.mha')
    Run('S:/Tristan/Mask/20201207_PDX_PAR1059x2GFP_100x15um_mask.mha')
    Run('S:/Tristan/Mask/20200908_PDX_PAR1022_gfp_100x15um_set1_mask.mha')
    Run('S:/Tristan/Mask/20201103_PDX_PAR1022_GFP_Rep2_100x15um_mask.mha')
    Run('S:/Tristan/Mask/20210119_PDX_STG316_GFP_100x15um_mask.mha')
    Run('S:/Tristan/Mask/20210121_PDX_STG316_GFP_001389_100x15um_mask.mha')

    Run('S:/Tristan/Mask/20210104_PDX_AB580Sc_GFP_100x15um_mask.mha')
    Run('S:/Tristan/Mask/20201216_PDX_STG143_SC_GFP_100x15um_mask.mha')
    Run('S:/Tristan/Mask/20201109_PDX_STG1394_GFP_100x15um_mask.mha')
    Run('S:/Tristan/Mask/20201110_PDX_HC1010_GFP_100x15um_mask.mha')
    Run('S:/Tristan/Mask/20210211_PDX_PAR1006_GFP_100x15um.mha')
    Run('S:/Tristan/Mask/20210127_PDX_AB630_GFP_100x15um_mask.mha')
    Run('S:/Tristan/Mask/20210126_AB764_017377_100x15um_gfp_mask.mha')


    

if False:
    import argparse
    parser = argparse.ArgumentParser(description='Quantify Stroma.')
    parser.add_argument('file_name')
    args = parser.parse_args()

    Run(args.file_name)

else:
    SegmentStroma()
