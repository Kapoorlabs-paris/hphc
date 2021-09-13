#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:50:47 2021

@author: vkapoor
"""
from tifffile import imread, imwrite
import csv
import napari
import glob
import os
import sys
import numpy as np
import json
from pathlib import Path
from scipy import spatial
import itertools
from napari.qt.threading import thread_worker
import matplotlib.pyplot  as plt
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QPushButton, QSlider
import h5py
from hphc.helpers import AfterUNET, Integer_to_border, MeasureArea, DistWater
from tifffile import imread, imwrite
import pandas as pd
import imageio
from tqdm import tqdm
import seaborn as sns
from skimage import measure
from skimage.measure import label
from dask.array.image import imread as daskread
Boxname = 'ImageIDBox'



class VizCorrect(object):

        def __init__(self, imagedir, savedir, fileextension = '*tif', hair_seg_name = 'Water', binary_name = 'BinaryWater'
                        , vein_name = 'Vein', mask_name = 'Mask', marker_name = 'Markers', doCompartment = False, max_size = 100000):
            
            
               self.imagedir = imagedir
               self.savedir = savedir
               self.fileextension = fileextension
               self.hair_seg_name = hair_seg_name
               self.binary_name = binary_name
               self.vein_name = vein_name
               self.mask_name = mask_name
               self.marker_name = marker_name 
               self.doCompartment = doCompartment 
               self.max_size = max_size 
               Path(self.savedir).mkdir(exist_ok=True)
               
               
               
               
        def load_json(self):
            with open(self.categories_json, 'r') as f:
                return json.load(f)      
            
            
        def showsecondNapari(self):
                 
                 self.viewer = napari.Viewer()
                 Raw_path = os.path.join(self.imagedir, self.fileextension)
                 X = glob.glob(Raw_path)
                 Imageids = []
                 
                 for imagename in X:
                     Imageids.append(imagename)
                     
                 
                
                    
                 imageidbox = QComboBox()   
                 imageidbox.addItem(Boxname)   
                 measurestatsbutton = QPushButton(' Measure Stats')
                 savebutton = QPushButton(' Save Images')
                    
                 for i in range(0, len(Imageids)):
                     
                     
                     imageidbox.addItem(str(Imageids[i]))
                     
                     
                 self.figure = plt.figure(figsize=(4, 4))
                 self.multiplot_widget = FigureCanvas(self.figure)
                 self.ax = self.multiplot_widget.figure.subplots(1, 1)
                 width = 400
                 dock_widget = self.viewer.window.add_dock_widget(
                 self.multiplot_widget, name="AreaStats", area='right')
                 self.multiplot_widget.figure.tight_layout()
                    
                 
                 
               
                 
                 imageidbox.currentIndexChanged.connect(
                 lambda trackid = imageidbox: self.second_image_add(
                         
                         imageidbox.currentText(),
                         
                         os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                         False,
                         False
                    
                )
            )            
                 
                 savebutton.clicked.connect(
                 lambda trackid = imageidbox: self.second_image_add(
                         
                         imageidbox.currentText(),
                         
                         os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                         False,
                         True
                    
                )
            )   
                    
                 measurestatsbutton.clicked.connect(
                 lambda trackid = imageidbox: self.second_image_add(
                         
                         imageidbox.currentText(),
                         
                         os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                         True,
                         False
                    
                )
            )    
                 
                 self.viewer.window.add_dock_widget(imageidbox, name="Image", area='bottom') 
                 self.viewer.window.add_dock_widget(savebutton, name="Save Images", area='bottom') 
                 self.viewer.window.add_dock_widget(measurestatsbutton, name="Measure Stats", area='bottom') 
                
        def showNapari(self):
                 
                 self.viewer = napari.Viewer()
                 Raw_path = os.path.join(self.imagedir, self.fileextension)
                 X = glob.glob(Raw_path)
                 Imageids = []
                 
                 for imagename in X:
                     Imageids.append(imagename)
                
                    
                 imageidbox = QComboBox()   
                 imageidbox.addItem(Boxname)   
                 detectionsavebutton = QPushButton(' Save Corrections')
                 
                    
                 for i in range(0, len(Imageids)):
                     
                     
                     imageidbox.addItem(str(Imageids[i]))
                     
                     
                 
                 imageidbox.currentIndexChanged.connect(
                 lambda trackid = imageidbox: self.image_add(
                         
                         imageidbox.currentText(),
                         
                         os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                     
                         False
                    
                )
            )            
                 
                 detectionsavebutton.clicked.connect(
                 lambda trackid = imageidbox: self.image_add(
                         
                         imageidbox.currentText(),
                         
                         os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                     
                         True
                    
                )
            )   
                 
                 self.viewer.window.add_dock_widget(imageidbox, name="Image", area='bottom') 
                 self.viewer.window.add_dock_widget(detectionsavebutton, name="Save Corrections", area='bottom')  
                 
                 
                        
        

                                     
        def second_image_add(self, image_toread, imagename, compute = False, save = False):
                
                if not compute and not save:
                        for layer in list(self.viewer.layers):

                            if 'Image' in layer.name or layer.name in 'Image':

                                                            self.viewer.layers.remove(layer)


                        self.image = daskread(image_toread)
                        if len(self.image.shape) > 3:
                            self.image = self.image[0,:]

                        self.maskimage = imread(self.savedir + imagename + self.mask_name + '.tif')
                        self.integerimage = imread(self.savedir + imagename + self.hair_seg_name + '.tif')
                        self.integerimage = remove_big_objects(self.integerimage, self.max_size)
                        self.binaryimage = imread(self.savedir + imagename + self.binary_name + '.tif')


                        self.viewer.add_image(self.image, name='Image'+imagename)
                        self.viewer.add_labels(self.integerimage, name ='Image'+'Integer_Labels'+imagename)

                        self.viewer.add_labels(self.binaryimage, name ='Image'+'Binary_Segmentation'+imagename)
                        self.viewer.add_labels(self.maskimage, name ='Image'+'Wing_Mask'+imagename)
                
                
                if compute:
                    
                    

                    ModifiedArraySeg = self.viewer.layers['Image'+'Integer_Labels'+imagename].data 
                    ModifiedArraySeg = ModifiedArraySeg.astype('uint16')
                    LabelMaskImage = ModifiedArraySeg > 0
                    Compartment = label(LabelMaskImage)
                    ModifiedArrayMask = self.viewer.layers['Image'+'Wing_Mask'+imagename].data 
                    ModifiedArrayMask = ModifiedArrayMask.astype('uint8')

                  
                    
                    self.dataset = MeasureArea(ModifiedArraySeg,LabelMaskImage, self.savedir, imagename, self.doCompartment)
                    self.dataset_index = self.dataset.index
                    self.ax.cla()
                    
                    self.ax.plot.hist(self.dataset.Area, density = True)
                    self.ax.set_title(imagename + "Size")
                    self.ax.set_xlabel("Time")
                    self.ax.set_ylabel("Counts")
                    self.figure.canvas.draw()
                    self.figure.canvas.flush_events()

                if save:

                        ModifiedArraySeg = self.viewer.layers['Image'+'Integer_Labels' + imagename].data 
                        ModifiedArraySeg = ModifiedArraySeg.astype('uint16')
                        LabelMaskImage = ModifiedArraySeg > 0
                        Compartment = label(LabelMaskImage)
                        ModifiedArrayMask = self.viewer.layers['Image'+'Wing_Mask'+imagename].data 
                        ModifiedArrayMask = ModifiedArrayMask.astype('uint8')

                        BinaryImage = Integer_to_border(ModifiedArraySeg)

                        imwrite((Resultsdir  +   Name + self.hair_seg_name + '.tif' ) , ModifiedArraySeg) 
                        imwrite((Resultsdir  +   Name + self.binary_name + '.tif' ) , BinaryImage)
                        imwrite((Resultsdir  +   Name + self.mask_name + '.tif' ) , MaskImage)

                                                  
            
        def image_add(self, image_toread, imagename, save = False):
                                    
               
                if not save:
                        for layer in list(self.viewer.layers):

                            if 'Image' in layer.name or layer.name in 'Image':

                                                            self.viewer.layers.remove(layer)
                        self.image = daskread(image_toread)
                        if len(self.image.shape) > 3:
                            self.image = self.image[0,:]

                        self.maskimage = imread(self.savedir + imagename + self.mask_name + '.tif')
                        self.integerimage = imread(self.savedir + imagename + self.hair_seg_name + '.tif')
                        self.integerimage = remove_big_objects(self.integerimage, self.max_size)
                        self.markerimage = imread(self.savedir + imagename + self.marker_name + '.tif')
                        self.markerimage = self.markerimage.astype('uint16')
                        self.viewer.add_image(self.image, name= 'Image'+imagename )

                        NewMarkerImage = np.zeros(self.markerimage.shape)

                        waterproperties = measure.regionprops(self.markerimage)

                        Coordinates = [prop.centroid for prop in waterproperties]

                        Coordinates = sorted(Coordinates , key=lambda k: [k[0], k[1]])
                        self.viewer.add_points(data = Coordinates, name='Image'+'Markers' +imagename, face_color='red', ndim = 2)


                if save:

                        NewCoordinates = self.viewer.layers['Image'+'Markers'].data  
                        NewMarkerImage[tuple(coordinates_int.T)] = 1 + np.arange(len(NewCoordinates))

                        markers = morphology.dilation(NewMarkerImage, morphology.disk(2))  

                        #Redo Watershed                     
                        WaterImage, BinaryImage = AfterUNET(self.integerimage > 0, NewCoordinates, self.maskimage, self.veinimage) 

                        imwrite((Resultsdir  +   Name + self.marker_name + '.tif' ) , markers) 
                        imwrite((Resultsdir  +   Name + self.hair_seg_name +  '.tif' ) , WaterImage) 
                        imwrite((Resultsdir  +   Name + self.binary_name + '.tif' ) , BinaryImage)
                                                    
                
                
def GetMarkers(image):
    
    
    MarkerImage = np.zeros([MarkerImage.shape])
    waterproperties = measure.regionprops(image)                
    Coordinates = [prop.centroid for prop in waterproperties]
    Coordinates = sorted(Coordinates , key=lambda k: [k[0], k[1]])
    MarkerImage[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))

    markers = morphology.dilation(MarkerImage, morphology.disk(2))        
   
    return markers 


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