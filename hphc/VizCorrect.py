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
import seaborn as sns
from skimage.measure import label
from dask.array.image import imread as daskread
Boxname = 'ImageIDBox'



class VizCorrect(object):

        def __init__(self, imagedir, savedir, fileextension = '*tif', hair_seg_name = 'Water', binary_name = 'BinaryWater'
                        , vein_name = 'Vein', mask_name = 'Mask', marker_name = 'Markers', doCompartment = False):
            
            
               self.imagedir = imagedir
               self.savedir = savedir
               self.fileextension = fileextension
               self.hair_seg_name = hair_seg_name
               self.binary_name = binary_name
               self.vein_name = vein_name
               self.mask_name = mask_name
               self.marker_name = marker_name 
               self.doCompartment = doCompartment 
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
                 measurestatsbutton = QPushButton(' Compute Statistics')
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
                 
                 self.viewer.window.add_dock_widget(imageidbox, name="Image", area='left') 
                 self.viewer.window.add_dock_widget(detectionsavebutton, name="Save Corrections", area='left')  
                 
                 
                        
        

                                     
        def second_image_add(self, image_toread, imagename, compute = False, save = False):
                                    
                for layer in list(self.viewer.layers):
                                         if 'Image' in layer.name or layer.name in 'Image':
                                                    self.viewer.layers.remove(layer)
                self.image = daskread(image_toread)
                if len(self.image.shape) > 3:
                    self.image = self.image[0,:]
                    
                self.maskimage = daskread(self.savedir + imagename + self.mask_name + '.tif')
                self.integerimage = daskread(self.savedir + imagename + self.hair_seg_name + '.tif')
                self.binaryimage = daskread(self.savedir + imagename + self.binary_name + '.tif')
                
                
                self.viewer.add_image(self.image, name= 'Image' + imagename )
                self.viewer.add_labels(self.integerimage, name = 'Integer_Labels')

                self.viewer.add_labels(self.binaryimage, name = 'Binary_Segmentation')
                self.viewer.add_labels(self.maskimage, name = 'Wing_Mask')
                
                
                if compute:
                    
                    
                    ModifiedArraySeg = viewer.layers['Integer_Labels'].data 
                    ModifiedArraySeg = ModifiedArraySeg.astype('uint16')
                    LabelMaskImage = ModifiedArraySeg > 0
                    Compartment = label(LabelMaskImage)
                    ModifiedArrayMask = viewer.layers['Wing_Mask'].data 
                    ModifiedArrayMask = ModifiedArrayMask.astype('uint8')

                    BinaryImage = Integer_to_border(ModifiedArraySeg)
                    
                    self.dataset, densityplot = MeasureArea(Compartment,LabelMaskImage, self.savedir, imagename, self.doCompartment)

                    self.dataset_index = self.dataset.index
                    self.ax.cla()
                    
                    self.ax.sns.histplot(self.dataset.Area, kde = True)
                    self.ax.set_title(imagename + "Size")
                    self.ax.set_xlabel("Time")
                    self.ax.set_ylabel("Counts")
                    self.figure.canvas.draw()
                    self.figure.canvas.flush_events()

                if save:

                        ModifiedArraySeg = viewer.layers['Integer_Labels'].data 
                        ModifiedArraySeg = ModifiedArraySeg.astype('uint16')
                        LabelMaskImage = ModifiedArraySeg > 0
                        Compartment = label(LabelMaskImage)
                        ModifiedArrayMask = viewer.layers['Wing_Mask'].data 
                        ModifiedArrayMask = ModifiedArrayMask.astype('uint8')

                        BinaryImage = Integer_to_border(ModifiedArraySeg)

                        imwrite((Resultsdir  +   Name + self.hair_seg_name + '.tif' ) , ModifiedArraySeg) 
                        imwrite((Resultsdir  +   Name + self.binary_name + '.tif' ) , BinaryImage)
                        imwrite((Resultsdir  +   Name + self.mask_name + '.tif' ) , MaskImage)

                                                  
            
        def image_add(self, image_toread, imagename, save = False):
                                    
                for layer in list(self.viewer.layers):
                                         if 'Image' in layer.name or layer.name in 'Image':
                                                    self.viewer.layers.remove(layer)
                self.image = daskread(image_toread)
                if len(self.image.shape) > 3:
                    self.image = self.image[0,:]
                    
                self.maskimage = daskread(self.savedir + imagename + self.mask_name + '.tif')
                self.integerimage = daskread(self.savedir + imagename + self.hair_seg_name + '.tif')
                self.veinimage = daskread(self.savedir + imagename + self.vein_name + '.tif')
                self.markerimage = daskread(self.savedir + imagename + self.marker_name + '.tif')
                
                
                self.viewer.add_image(self.image, name= 'Image' + imagename )
                
                NewMarkerImage = np.zeros(self.integerimage.shape)
                waterproperties = measure.regionprops(self.integerimage)
                Coordinates = [prop.centroid for prop in waterproperties]
                Coordinates = sorted(Coordinates , key=lambda k: [k[0], k[1]])
                viewer.add_points(data = Coordinates, name= 'Markers', face_color='red', ndim = 2)


                if save:

                        NewCoordinates = viewer.layers['Markers'].data  
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