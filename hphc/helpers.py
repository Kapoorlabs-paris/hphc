#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:08:41 2019

@author: aimachine
"""

from __future__ import print_function, unicode_literals, absolute_import, division
#import matplotlib.pyplot as plt
import numpy as np
import collections
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries
import warnings
from skimage.filters import gaussian
from six.moves import reduce
from matplotlib import cm
from skimage.filters import threshold_local, threshold_otsu
from skimage.morphology import remove_small_objects, thin
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
from scipy.spatial import Voronoi, voronoi_plot_2d
from skimage.segmentation import watershed
import os
from skimage.segmentation import find_boundaries
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import difflib
import pandas as pd
import glob
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from scipy.spatial import ConvexHull
from tifffile import imread, imwrite
from scipy import ndimage as ndi
from pathlib import Path
from tqdm import tqdm
from skimage.segmentation import  relabel_sequential
from skimage import morphology
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import  binary_dilation, binary_erosion
from skimage.util import invert 
from skimage import measure
from skimage.filters import sobel
from skimage.measure import label
from scipy import spatial
from skimage.draw import polygon
from shapely.geometry import MultiPoint, Point, Polygon
from csbdeep.data import  create_patches,create_patches_reduced_target, RawData
from skimage import transform



def _fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img==l
        mask_filled = binary_fill_holes(mask,**kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled
def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""
    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i,sl in enumerate(objects,1):
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask_filled = binary_fill_holes(grown_mask,**kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def dilate_label_holes(lbl_img, iterations):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (range(np.min(lbl_img), np.max(lbl_img) + 1)):
        mask = lbl_img==l
        mask_filled = binary_dilation(mask,iterations = iterations)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled


def remove_big_objects(ar, max_size=6400, connectivity=1, in_place=False):
    
    out = ar.copy()
    ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")



    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0

    return out
def intersection(lst1, lst2):
    print('list',lst1)
    print('list2', lst2)
    lst3 = set([tuple(sorted(ele)) for ele in lst1]) & set([tuple(sorted(ele)) for ele in lst2])
  
    return lst3


def DistWater(image, Coordinates, Mask, VeinMask, indices, maskindices):

    #distance = ndi.distance_transform_edt(np.logical_not(image))

    Mask = np.logical_xor(Mask > 0, VeinMask > 0)
    coordinates_int = np.round(Coordinates).astype(int)
    markers_raw = np.zeros(image.shape)  
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))

    markers = morphology.dilation(markers_raw, morphology.disk(2))
  
    Labelimage = watershed(image, markers)
    Labelimage = Remove_label(Labelimage, indices)
    Labelimage = Remove_label(Labelimage, maskindices)
    Binaryimage = Integer_to_border(Labelimage.astype('uint16'))
    binary = Integer_to_border(Labelimage.copy().astype('uint16'))
     
    return Labelimage, binary, markers
    

def voronoi_finite_polygons_2d(vor, indices, maskindices, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    vol = np.zeros(vor.npoints)
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        


        if -1 in vertices: # some regions can be opened
            vol[p1] = np.inf
        else:
            vol[p1] = ConvexHull(vor.vertices[vertices]).volume
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices), vol

def remove_big_objects(ar, max_size=6400, connectivity=1, in_place=False):
    
    out = ar.copy()
    ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0

    return out          


def ProjUNETPrediction(filesRaw, modelVein, modelHair, SavedirMax, SavedirAvg,SavedirVein, SavedirHair,  n_tiles, axis, DoVoronoi = False, DoWatershed = True,min_size = 20, sigma = 5, show_after = 1, scales = 10, maxsize = 10000):


    count = 0
    Path(SavedirMax).mkdir(exist_ok=True)
    Path(SavedirAvg).mkdir(exist_ok=True)
    Path(SavedirVein).mkdir(exist_ok=True)
    Path(SavedirHair).mkdir(exist_ok=True)
    for fname in filesRaw:
            count = count + 1
            
            Name = os.path.basename(os.path.splitext(fname)[0])
            image = imread(fname)
            maximage = np.max(image, axis = 0)
            avgimage = np.mean(image, axis = 0)
            Maskimage = BadSegmentation(maximage, min_size = min_size, sigma = sigma)
            imwrite(SavedirMax + Name + 'Mask' + '.tif', Maskimage.astype('uint8'))
            imwrite(SavedirMax + Name + '.tif', maximage.astype('uint8'))
            imwrite(SavedirAvg + Name + '.tif', avgimage.astype('uint8'))
            
            
            Hairimage = Segment(maximage, modelHair, axis, n_tiles, show_after =  show_after)
            Veinimage = Segment(avgimage, modelVein, axis, n_tiles, show_after =  show_after)

            BinaryVeinimage = Integer_to_border(Veinimage.astype('uint16'))

            Labelimage = np.zeros(Hairimage.shape)
            Veinimagecopy = Veinimage.copy()
            indices = np.where(Veinimagecopy > 0)
            Hairimage[indices] = 0
            
            Maskimagecopy = Maskimage.copy()
            maskindices = np.where(Maskimagecopy == 0)
            Hairimage[maskindices] = 0
           
            
            if count%show_after == 0:
                    doubleplot(Veinimage, Hairimage, "Vein", "Hair")
                    doubleplot(maximage, Maskimage, "Original", "Mask")
                    
            LabelHairimage = label(Hairimage)
            waterproperties = measure.regionprops(LabelHairimage, LabelHairimage)
            Coordinates = [prop.centroid for prop in waterproperties]
            Coordinates = sorted(Coordinates , key=lambda k: [k[0], k[1]])
            Coordinates.append((0,0))
            Coordinates = np.asarray(Coordinates)
            
            if DoWatershed:

               Hairimage[np.where(Hairimage > 0)] = 127
               Maskimage[np.where(Maskimage > 0)] = 255
               Hairimage = np.logical_xor(Maskimage, Hairimage)
               Hairimage = np.logical_xor(Hairimage , Veinimage)

               distlabel, distbinary, markers = DistWater(Hairimage, Coordinates, Maskimage, Veinimage, indices, maskindices)
               distlabel = remove_big_objects(distlabel, maxsize)
               distlabelrelabel = RelabelArea(distlabel, SavedirHair, Name, scales)
               if count%show_after == 0:
                   doubleplot(distlabel, distbinary, "Label Water", "Binary Water")
               imwrite(SavedirHair + Name + 'BinaryWater' + '.tif', distbinary.astype('uint8'))
               imwrite(SavedirHair + Name + 'Water' + '.tif', distlabel.astype('uint16'))
               imwrite(SavedirHair + Name + 'WaterRelabelArea' + '.tif', distlabelrelabel.astype('uint16'))
               imwrite(SavedirHair + Name + 'Markers' + '.tif', markers.astype('uint16'))
             
            if DoVoronoi:
                  vor = Voronoi(Coordinates)
                  regions, vertices, vol = voronoi_finite_polygons_2d(vor, indices, maskindices)
                  pts = MultiPoint([Point(i) for i in Coordinates])
                  mask = pts.convex_hull
                  labelindex = 1 
                  for i in range(len(regions)):
                      region = regions[i]
                      volume = vol[i]
                      if volume is not np.inf:
                              polygon = vertices[region]
                              shape = list(polygon.shape)
                              shape[0] += 1
                              p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
                              polyY = np.array(list(zip(p.boundary.coords.xy[0][:-1])))
                              polyX = np.array(list(zip(p.boundary.coords.xy[1][:-1])))

                              smalllabel = polygons_to_label_coord(polyY, polyX, maximage.shape, labelindex)
                              Labelimage = Labelimage + smalllabel
                              labelindex = labelindex + 1
                              
                  Labelimage = Remove_label(Labelimage, indices)
                  Labelimage = Remove_label(Labelimage, maskindices)

                  Labelimage = remove_big_objects(Labelimage.astype('uint16'), maxsize)
                  Labelimagerelabel = RelabelArea(Labelimage.astype('uint16'),SavedirHair, Name, scales)
                  Binaryimage = Integer_to_border(Labelimage.astype('uint16'))
                  if count%show_after == 0:
                      doubleplot(Labelimage, Binaryimage, "Label Voronoi", "Binary Voronoi")

                  imwrite(SavedirHair + Name + 'BinaryVor' + '.tif', Binaryimage.astype('uint8'))
                  imwrite(SavedirHair + Name + 'Vor' + '.tif', Labelimage.astype('uint16'))
                  imwrite(SavedirHair + Name + 'VorRelabelArea' + '.tif', Labelimagerelabel.astype('uint16'))
            
            imwrite(SavedirVein + Name + '.tif', Veinimage.astype('uint16'))
            imwrite(SavedirHair + Name + '.tif', Hairimage.astype('uint16'))
            


        
        
    

def Integer_to_border(Label):

        BoundaryLabel =  find_boundaries(Label, mode='outer')
           
        Binary = BoundaryLabel > 0
        
        return Binary


def Remove_label(Label, indices):
  
    Label[indices] = 0 
    return Label    


def RelabelArea(Label, SavedirHair, Name,  scale):

     regions = measure.regionprops(Label)
     areas = [int(regions[i].area) for i in range(len(regions))]
     Relabel = np.zeros(Label.shape)
     minArea = min(areas)
     maxArea = max(areas)
     print(minArea, maxArea)
     scalearea = [minArea + (t/scale) * (maxArea - minArea) for t in range(0,scale)]
     scalearea = np.round(scalearea).astype(int)
     scalearea = np.asarray(scalearea)
     print(scalearea)
     Label_ids = []
     Area_ids = []
     
     for i in range(len(regions)):
        label_id = regions[i].label
        area_id = int(regions[i].area)
        Label_ids.append(label_id)
        Area_ids.append(area_id)
        
        scale_id = min(scalearea, key=lambda x:abs(x-area_id))
        only_current_label_id = np.where(Label == label_id, scale_id, 0)
        Relabel = Relabel + only_current_label_id
        
     df = pd.DataFrame(list(zip(Label_ids,Area_ids)), 
                                                                      columns =['Label_ID', 'Area'])
     
     mean_area = df['Area'].mean()
     max_area = df['Area'].max()
     max_area_index = df.index[df['Area'] == max_area]
     max_label = df.at[max_area_index, 'Label_ID']
     
     print('Mean Area', mean_area, 'Max Area', max_area, 'Max Area Label', max_label)
     
     df.to_csv(SavedirHair + '/' + Name + 'Area_Stats' +  '.csv')
        
     return Relabel   

def polygons_to_label_coord(Y, X, shape, labelindex):
    """renders polygons to image of given shape
    """

    lbl = np.zeros(shape,np.int32)
    rr,cc = polygon(Y, X)
    lbl[rr,cc] = labelindex

    return lbl

def swap(*line_list):
    """
    Example
    -------
    line = plot(linspace(0, 2, 10), rand(10))
    swap(line)
    """
    for lines in line_list:
        try:
            iter(lines)
        except:
            lines = [lines]

        for line in lines:
            xdata, ydata = line.get_xdata(), line.get_ydata()
            line.set_xdata(ydata)
            line.set_ydata(xdata)
            line.axes.autoscale_view()        
def voronoi(points,shape=(500,500)):
    depthmap = np.ones(shape,np.float)*1e308
    colormap = np.zeros(shape,np.int)

    def hypot(X,Y):
        return (X-x)**2 + (Y-y)**2

    for i,(x,y) in enumerate(points):
        paraboloid = np.fromfunction(hypot,shape)
        colormap = np.where(paraboloid < depthmap,i+1,colormap)
        depthmap = np.where(paraboloid <
depthmap,paraboloid,depthmap)

    for (x,y) in points:
        colormap[x-1:x+2,y-1:y+2] = 0

    return colormap

def draw_map(colormap):
    shape = colormap.shape

    palette = np.array([
            0x000000FF,
            0xFF0000FF,
            0x00FF00FF,
            0xFFFF00FF,
            0x0000FFFF,
            0xFF00FFFF,
            0x00FFFFFF,
            0xFFFFFFFF,
            ])

    colormap = np.transpose(colormap)
    pixels = np.empty(colormap.shape+(4,),np.int8)

    pixels[:,:,3] = palette[colormap] & 0xFF
    pixels[:,:,2] = (palette[colormap]>>8) & 0xFF
    pixels[:,:,1] = (palette[colormap]>>16) & 0xFF
    pixels[:,:,0] = (palette[colormap]>>24) & 0xFF

    return Image
def expand_labels(label_image, distance=1):
    """Expand labels in label image by ``distance`` pixels without overlapping.
    Given a label image, ``expand_labels`` grows label regions (connected components)
    outwards by up to ``distance`` pixels without overflowing into neighboring regions.
    More specifically, each background pixel that is within Euclidean distance
    of <= ``distance`` pixels of a connected component is assigned the label of that
    connected component.
    Where multiple connected components are within ``distance`` pixels of a background
    pixel, the label value of the closest connected component will be assigned (see
    Notes for the case of multiple labels at equal distance).
    Parameters
    ----------
    label_image : ndarray of dtype int
        label image
    distance : float
        Euclidean distance in pixels by which to grow the labels. Default is one.
    Returns
    -------
    enlarged_labels : ndarray of dtype int
        Labeled array, where all connected regions have been enlarged
    Notes
    -----
    Where labels are spaced more than ``distance`` pixels are apart, this is
    equivalent to a morphological dilation with a disc or hyperball of radius ``distance``.
    However, in contrast to a morphological dilation, ``expand_labels`` will
    not expand a label region into a neighboring region.  
    This implementation of ``expand_labels`` is derived from CellProfiler [1]_, where
    it is known as module "IdentifySecondaryObjects (Distance-N)" [2]_.
    There is an important edge case when a pixel has the same distance to
    multiple regions, as it is not defined which region expands into that
    space. Here, the exact behavior depends on the upstream implementation
    of ``scipy.ndimage.distance_transform_edt``.
    See Also
    --------
    :func:`skimage.measure.label`, :func:`skimage.segmentation.watershed`, :func:`skimage.morphology.dilation`
    References
    ----------
    .. [1] https://cellprofiler.org
    .. [2] https://github.com/CellProfiler/CellProfiler/blob/082930ea95add7b72243a4fa3d39ae5145995e9c/cellprofiler/modules/identifysecondaryobjects.py#L559
    Examples
    --------
    >>> labels = np.array([0, 1, 0, 0, 0, 0, 2])
    >>> expand_labels(labels, distance=1)
    array([1, 1, 1, 0, 0, 2, 2])
    Labels will not overwrite each other:
    >>> expand_labels(labels, distance=3)
    array([1, 1, 1, 1, 2, 2, 2])
    In case of ties, behavior is undefined, but currently resolves to the
    label closest to ``(0,) * ndim`` in lexicographical order.
    >>> labels_tied = np.array([0, 1, 0, 2, 0])
    >>> expand_labels(labels_tied, 1)
    array([1, 1, 1, 2, 2])
    >>> labels2d = np.array(
    ...     [[0, 1, 0, 0],
    ...      [2, 0, 0, 0],
    ...      [0, 3, 0, 0]]
    ... )
    >>> expand_labels(labels2d, 1)
    array([[2, 1, 1, 0],
           [2, 2, 0, 0],
           [2, 3, 3, 0]])
    """

    distances, nearest_label_coords = distance_transform_edt(
        label_image == 0, return_indices=True
    )
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out          
def Integer_to_border(Label):

        BoundaryLabel =  find_boundaries(Label, mode='outer')
           
        Binary = BoundaryLabel > 0
        
        return Binary
def Segment(image, model, axis, n_tiles, show_after =  1 ):
    
            Segmented = model.predict(image, axis, n_tiles = n_tiles)
            thresh = threshold_otsu(Segmented)
            Binary = Segmented > thresh
            
            
            return Binary


    
def generate_2D_patch_training_data(BaseDirectory, SaveNpzDirectory, SaveName, patch_size = (512,512), n_patches_per_image = 64, transforms = None):

    
    raw_data = RawData.from_folder (
    basepath    = BaseDirectory,
    source_dirs = ['Original'],
    target_dir  = 'BinaryMask',
    axes        = 'YX',
    )
    
    X, Y, XY_axes = create_patches (
    raw_data            = raw_data,
    patch_size          = patch_size,
    n_patches_per_image = n_patches_per_image,
    transforms = transforms,
    save_file           = SaveNpzDirectory + SaveName,
    )


     
   
def BadSegmentation(maximage, min_size = 20, sigma = 5):

                    maximage = gaussian_filter(maximage, sigma = sigma)
                    thresh = threshold_otsu(maximage) 
                    maximage = maximage > thresh
                    maximage = label(maximage)
                    maximage = remove_small_objects(maximage, min_size)
                    maximage = maximage > 0
                    maximage = fill_label_holes(maximage) 
                    return maximage     

        

def Label_counter(filesRaw, ProbabilityThreshold, Resultdir, min_size = 10 ):


     AllCount = []
     AllName = []
     for fname in filesRaw:
        Name = os.path.basename(os.path.splitext(fname)[0])
        TwoChannel = imread(fname)
        SpotChannel = TwoChannel[:,0,:,:]
        Binary = SpotChannel > ProbabilityThreshold
        Binary = remove_small_objects(Binary, min_size = min_size)
        Integer = label(Binary)
        waterproperties = measure.regionprops(Integer, Integer)
        labels = []
        for prop in waterproperties:
            if prop.label > 0:
                     
                      labels.append(prop.label)
        count = len(labels)
        imwrite(Resultdir + Name + '.tif', Integer.astype('uint16')) 
        AllName.append(Name)
        AllCount.append(count)
        
     df = pd.DataFrame(list(zip(AllCount)), index =AllName, 
                                                  columns =['Count'])
    
     df.to_csv(Resultdir + '/' + 'CountMasks' +  '.csv')  
     df     
    
    



def multiplot(imageA, imageB, imageC, titleA, titleB, titleC, targetdir = None, File = None, plotTitle = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].imshow(imageA, cmap=cm.gray)
    ax[0].set_title(titleA)
    ax[0].set_axis_off()
    ax[1].imshow(imageB, cmap=plt.cm.nipy_spectral)
    ax[1].set_title(titleB)
    ax[1].set_axis_off()
    ax[2].imshow(imageC, cmap=plt.cm.nipy_spectral)
    ax[2].set_title(titleC)
    ax[2].set_axis_off()
    plt.tight_layout()
    plt.show()
    for a in ax:
      a.set_axis_off()
      
def doubleplot(imageA, imageB, titleA, titleB, targetdir = None, File = None, plotTitle = None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].imshow(imageA, cmap=cm.gray)
    ax[0].set_title(titleA)
    ax[0].set_axis_off()
    ax[1].imshow(imageB, cmap=plt.cm.nipy_spectral)
    ax[1].set_title(titleB)
    ax[1].set_axis_off()

    plt.tight_layout()
    plt.show()
    for a in ax:
      a.set_axis_off() 




