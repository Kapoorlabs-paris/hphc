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
import difflib
import pandas as pd
import glob
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



            

def ProjUNETPrediction(filesRaw, modelVein, modelHair, SavedirMax, SavedirAvg,SavedirVein, SavedirHair,  n_tiles, axis,min_size = 20, sigma = 5, show_after = 1):

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
          
            Veinimagecopy = Veinimage.copy()
            indices = np.where(Veinimagecopy > 0)
            Hairimage[indices] = 0
            
            Maskimagecopy = Maskimage.copy()
            maskindices = np.where(Maskimagecopy > 0)
            Hairimage[maskindices] = 0
            
            if count%show_after == 0:
                    doubleplot(Veinimage, Hairimage, "Vein", "Hair")
                    doubleplot(maximage, Maskimage, "Original", "Mask")
                    
            LabelHairimage = label(Hairimage)
            waterproperties = measure.regionprops(LabelHairimage, LabelHairimage)
            Coordinates = [prop.centroid for prop in waterproperties]
            Coordinates = sorted(Coordinates , key=lambda k: [k[1], k[0]])
            Coordinates.append((0,0))
            Coordinates = np.asarray(Coordinates)

            coordinates_int = np.round(Coordinates).astype(int)
            markers_raw = np.zeros_like(Maskimage)  
            markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))

            markers = morphology.dilation(markers_raw, morphology.disk(2))
            watershedImage = watershed(Maskimage, markers)
            watershedImage = Integer_to_border(watershedImage)
            plt.imshow(watershedImage)
            plt.show()
            imwrite(SavedirVein + Name + '.tif', Veinimage.astype('uint16'))
            imwrite(SavedirHair + Name + '.tif', Hairimage.astype('uint16'))
            imwrite(SavedirHair + Name + 'vor' + '.tif', watershedImage.astype('uint16'))

          
def Integer_to_border(Label):

        BoundaryLabel =  find_boundaries(SmallLabel, mode='outer')
           
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
                    maximage = invert(maximage)
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




