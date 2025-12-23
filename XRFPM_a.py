# -*- coding: utf-8 -*-
"""
Created on Mon May  5 10:07:37 2025

@author: lucad
"""


""" 
Imports
"""
import os
import sys
import time

pc = "Pedro" if os.path.exists('C:/Users/Pedro/OneDrive - UGent/master 2/thesis/code/XProc-main') else "lucad"

sys.path.insert(1, 'C:/Users/' + pc + '/OneDrive - UGent/master 2/thesis/code/XProc-main')


# import skimage as sk
import skimage.transform as skt
import tifffile as tf
import pandas as pd
# import XProc as xpr
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import tomopy
from PIL import Image
import xraylib as xrl
import h5py



"""
Classes
"""

class mineral_phase(object):
    def __init__(self, name, density, major_composition):
        """
        initialise mineral_phase class

        Parameters
        ----------
        name : TYPE, str
            DESCRIPTION: name of mineral phase
        density : TYPE, float
            DESCRIPTION: density of mineral phase in g/cm3
        major_composition : TYPE, list(str)
            DESCRIPTION: list of bruto formula for mineral phase

        Returns
        -------
        None.

        """
        
        self.name = name #name of the mineral
        self.density = density #average density of the mineral
        self.major_composition = major_composition #major consitituents
        self.atten_range = []
        self.composition = ['unknown']
        self.atten_range = []

    def __str__(self):
        return f'{self.name}'
    

    def density(self):
        return self.density
    
    def name(self):
        return self.name
    
    def add_composition(self, composition):
        self.composition = composition
        return self.composition
    
    def add_attenuation_range(self, atten_range):
        self.atten_range = atten_range
        return self.atten_range
    
    def major_compositions(self):
        return self.major_composition
    

class pixels(object):
    
    def __init__(self, index, values, mineral_phase, shape, Ps_cor):
        """
        initialise pixels class

        Parameters
        ----------
        index : TYPE, int
            DESCRIPTION: index of pixel
        values : TYPE, 2d array
            DESCRIPTION: 2d array of original image values
        mineral_phase : TYPE, mineral_phase class
            DESCRIPTION: mineral phase of pixel, based on linear attenuation coefficient
        shape : TYPE, tuple(int)
            DESCRIPTION: tuple of values shape
        Ps_cor : TYPE, float
            DESCRIPTION: gives the correlation between a pixel and its size in real life

        Returns
        -------
        None.

        """
        
        self.index = index # index of the pixel, row by row starting from 0
        self.values = values # 2d array of values
        self.phase = mineral_phase #mineral phase of the pixel
        self.shape = shape #shape of the pixel
        self.conc = mineral_phase.composition #concentration of each element, used for iterative quantification
        self.cor = Ps_cor # pixel to real size correlation


    def __str__(self):
        return f'pixel nr. {self.index} has following values: {self.values}, which correlates to {self.phase} mineral phase. Each pixel represents {self.cor} (unit)'


    def index(self):
        return self.index
    
    def values(self):
        return self.values

    def phase(self):
        return self.phase
    
    def shape(self):
        return self.shape
    
    def cor(self):
        return self.cor
    
    def conc(self):
        return self.conc
        

class voxels(object):
    
    def __init__(self, index, values, mineral_phase, shape, Ps_cor):
        """
        initialise pixels class

        Parameters
        ----------
        index : TYPE, int
            DESCRIPTION: index of pixel
        values : TYPE, 3d array
            DESCRIPTION: 3d array of original image values
        mineral_phase : TYPE, mineral_phase class
            DESCRIPTION: mineral phase of pixel, based on linear attenuation coefficient
        shape : TYPE, tuple(int)
            DESCRIPTION: tuple of values shape
        Ps_cor : TYPE, float
            DESCRIPTION: gives the correlation between a voxel and its size in real life

        Returns
        -------
        None.

        """
        
        self.index = index # index of the pixel, row by row starting from 0
        self.values = values # 3d array of values
        self.phase = mineral_phase #mineral phase of the pixel
        self.shape = shape #shape of the pixel
        self.conc = mineral_phase.composition #concentration of each element, used for iterative quantification
        self.cor = Ps_cor # pixel to real size correlation


    def __str__(self):
        return f'voxel nr. {self.index} has following values: {self.values}, which correlates to {self.phase} mineral phase. Each pixel represents {self.cor} (unit)'


    def index(self):
        return self.index
    
    def values(self):
        return self.values

    def phase(self):
        return self.phase
    
    def shape(self):
        return self.shape
    
    def cor(self):
        return self.cor




""" 
Functions for CT creation
"""

def get_metadata(image):
    """
    Gets the metadata from a tiff image file

    Parameters
    ----------
    image : tiff image file
        print the metadata from the tiff file

    Returns
    -------
    None.

    """
    #open image using tifffile and get all possible metadata
    with tf.TiffFile(image) as f:
        metadata = f.pages[0].tags
        for tag in metadata.values():
            print(tag.name, tag.value)
    
    return None


def ct_scan(path_ct=None, image=None, array=None):
    """
    Convert a tif/tiff image to a 2d np array

    Parameters
    ----------
    path_ct : TYPE, str
        DESCRIPTION: path to where image file is stored
    image : TYPE, str
        DESCRIPTION: image file name
    array : TYPE, 2d list
        DESCRIPTION: optional parameter for code tests, DEFAULT is None

    Returns
    -------
    img_array : TYPE, 2d np array
        DESCRIPTION: 2d np array of image values

    """
    #open image and convert to 2d numpy array
    img = Image.open(path_ct+image) if path_ct and image != None else None
    img_array = np.array(img) if img != None else array
    
    return img_array


def display_ct_scan(path_ct=None, image=None, array=None):
    """
    Displays a tif/tiff image in grayscale

    Parameters
    ----------
    path_ct : TYPE, str
        DESCRIPTION: path to where image file is stored
    image : TYPE, str
        DESCRIPTION: image file name
    array : TYPE, 2d list
        DESCRIPTION: optional parameter for code tests, DEFAULT is None

    Returns
    -------
    None.

    """
    #convert image to 2d np array
    img_array= ct_scan(path_ct, image, array)
    
    #plot 2d array grayscaled
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    plt.imshow(img_array)
    ax.set_aspect('equal')
    
    plt.axis("off")
    # plt.style.use('grayscale')
    plt.show()
    
    return None


def ct_scans(path_ct=None, images=None,arrays=None):
    """
    Convert multiple ct scans (tif/tiff images) to 2d np arrays

    Parameters
    ----------
    path_ct : TYPE, str
        DESCRIPTION: path to where image files are stored
    images : TYPE, list(str)
        DESCRIPTION. list of image file names
    arrays : TYPE, 3d list
        DESCRIPTION. optional parameter for code testing, DEFAULT IS None

    Returns
    -------
    ims : TYPE, 3d np array 
        3d np array of ct scans 

    """
    #open images and convert each to 2d numpy array
    ims = []
    if path_ct and images != None:
        
        for im in images:
            img = Image.open(path_ct+im)
            img_array = np.array(img)
            ims.append(img_array)
    else:
        ims = arrays
        
    return ims


def display_ct_scans(path_ct=None, images=None):
    """
    Displays multiple ct scans in grayscale 

    Parameters
    ----------
    path_ct : TYPE, str
        DESCRIPTION: path to where the image files are stored
    images : TYPE, list(str)
        DESCRIPTION: list of names of image files (tif/tiff)

    Returns
    -------
    None.

    """
    #displays multiple ct scans akin to display_ct_scan
    for im in images:
        display_ct_scan(path_ct, im)
        
    return None


def add_ct_scan(filename,path_file,image_name, path_ct=None, image= None, array=None):
    """
    Writes/creates ct scan in h5 file

    Parameters
    ----------
    filename : TYPE, str
        DESCRIPTION: name of h5 to create/write in
    path_file : TYPE, str
        DESCRIPTION: path where h5 file should be found/created
    image_name : TYPE, str
        DESCRIPTION: name that image should hold in h5 file
    path_ct : TYPE, str
        DESCRIPTION: path to where image file is stored
    image : TYPE, str
        DESCRIPTION: image file name
    array : TYPE, 2d list
        DESCRIPTION: optional parameter to test code, DEFAULT is None

    Returns
    -------
    None.

    """
    #open and convert ct scan to 2d np array
    image = ct_scan(path_ct=path_ct, image=image, array=array)
    
    #write 2d array to h5 file
    with h5py.File(path_file + filename+".h5", 'r+') as f:
        try:
            f.create_group("ct scans")
            
        except Exception:
            pass
        f['ct scans'].create_dataset(image_name, data=image_name)
        f["ct scans"].create_dataset("scan", data=image)
    
    return None
    

def add_ct_scans(filename, path_file, images_names, path_ct=None, images=None, arrays=None):
    """
    Add multiple ct scans to h5 file

    Parameters
    ----------
    filename : TYPE, str
        DESCRIPTION: name of h5 file that should be created/written in
    path_file : TYPE, str
        DESCRIPTION: path to where h5 should be stored/created
    images_names : list(str)
        DESCRIPTION: list of image names that should be written in file
    path_ct : TYPE, str
        DESCRIPTION: path to where image files are located
    images : TYPE, list(str)
        DESCRIPTION. list of image file names to write in h5 file
    arrays : TYPE, 3d list
        DESCRIPTION: optional parameter to test code, DEFAULT is None

    Returns
    -------
    None.

    """
    #open and convert ct scans to 2d arrays
    images = ct_scans(path_ct=path_ct,images=images, arrays=arrays)
    ctscans = [im for im in images]
    
    #write 2d arrays to h5 file
    with h5py.File(path_file+filename+".h5", 'r+') as f:
        try:
            f.create_group("ct scans")
        
        except Exception:
            pass 
            
        f['ct scans'].create_dataset("names", data = images_names)
        f['ct scans'].create_dataset("scans", data = ctscans)
            
    return None




"""
functions for pixelization
"""

def pixelisation(path='', image='', array=[], cols=2, rows=2):
    """
    Divide image or 2d array into pixels of chosen size

    Parameters
    ----------
    path : TYPE, str
        DESCRIPTION: path to where image file is stored
    image : TYPE, str
        DESCRIPTION: file name of to be opened image
    array : TYPE, 2d list
        DESCRIPTION: optional parameter to test code, DEFAULT is list()
    cols : TYPE, int
        DESCRIPTION: amount of columns the pixels should be in size, DEFAULT is 2
    rows : TYPE, int
        DESCRIPTION: amount of rows the pixels should be in size, DEFAULT is 2

    Returns
    -------
    TYPE, 3d array
        3d array of pixels(2d arrays)

    """
    #open image and convert to 2d array
    im = Image.open(path+image) if path and image != '' else None
    img_array = np.array(im) if im != None else array
    
    #check if possible to divide image into chosen pixel size
    try:
        assert len(img_array) % rows == 0 and len(img_array) // rows != 0, "Error: image size cannot be properly divided into chosen row size"
        assert len(img_array[0]) % cols == 0 and len(img_array[0]) // cols != 0, "Error: image size cannot be properly divided into chosen column size"
    
    except AssertionError as error:
        print(error)
    
    #calculate amount of sections for chosen pixel size
    cols_ind, rows_ind = len(img_array[0]) // cols, len(img_array) // rows
    
    #divide original 2d array into 4d array
    columns =  np.array(np.hsplit(img_array, cols_ind))
    pix = np.array(np.hsplit(columns, rows_ind))
    
    #rearrange created 4d array into appropriate 3d array
    pixels = [pix[i][j] for i in range(rows_ind) for j in range(cols_ind)]
    return np.array(pixels)


#size_cols = size original image
def rejoin(pixels, size_cols):
    """
    Rejoins 3d array of pixels (2d array) into original 2d array 

    Parameters
    ----------
    pixels : TYPE, 3d array(2d array)
        DESCRIPTION: 3d array of pixels given from pixelisation function
    size_cols : TYPE, int
        DESCRIPTION: amount of columns of original image

    Returns
    -------
    TYPE, 2d array
        original 2d array of image values

    """
    
    #generate 2d array of original image size
    pix_full = np.prod(pixels.shape) #total amount of data points # can be written as pixels.size
    rows = pix_full // size_cols # returns the amount of rows given a rectangular image given
    full = [ [] for i in range(rows)] #create 2d array with correct amount of rows
    row, count = 0, 0
    for i in pixels:
        
        pixel_size = i.shape # first is amount per rows 2nd is amount per columns
        
        #append correct voxel to correct original row
        for n, k in enumerate(i):
            full[n+row] = np.concatenate((full[n+row], k), axis=None)
        count += 1
    
        if count * pixel_size[1] == size_cols:
            row += pixel_size[0]
            count = 0
    
    return np.array(full)



def rejoin_map_ct(Pixels, size_cols, viridis=False):
    """
    Rejoins dictionary(pixelclass) pixels (2d array) into original 2d array with mineral phases as intensities 

    Parameters
    ----------
    pixels : TYPE, dict(index:pixelclass)
        DESCRIPTION: dictionary of pixelclass akin to those create by create_pixels
    size_cols : TYPE, int
        DESCRIPTION: amount of columns of original image
    viridis  :  TYPE, boolean
        DESCRIPTION: if colormap should be in viridis scale or not, DEFAULT is FALSE

    Returns
    -------
    None.

    """
    #remove dummy pixeltype -1
    Pixels = dict(list(Pixels.items())[:-1])
    
    
    #pixels is dict pixel class, size_cols is original tiff image size
    pix_full = np.prod(Pixels[0].shape)*len(Pixels)
    rows = pix_full // size_cols #aount of rows given a given image
    full = [[] for i in range(rows)]#create 2d array with correct amount of rows
    row, count = 0, 0
    phases = np.array([[[Pixels[i].phase for _ in range(Pixels[i].shape[1])] for _ in range(Pixels[i].shape[0])] for i in Pixels])

    for i in phases:
        
        pixel_size = i.shape # first is amount per rows 2nd is amount per columns
        
        #append correct voxel to correct original row
        for n, k in enumerate(i):
            full[n+row] = np.concatenate((full[n+row], k), axis=None)
        count += 1
    
        if count * pixel_size[1] == size_cols:
            row += pixel_size[0]
            count = 0
            
    full = np.array(full) #2d array of mineral phases
    
    #set colormapping
    phases = list({phase for phase in np.ravel(full)})
    phaseslabels = {phase : i for i, phase in enumerate(phases)}
    full_int = np.vectorize(phaseslabels.get)(full)
    if viridis is True:
        int_unique_categories = len(np.unique(full_int))
        cmap = plt.get_cmap('viridis', int_unique_categories)
    
    else:
        cmap = 'tab10'
    #plot image with calculated colormapping
    plt.imshow(full_int, cmap=cmap)
    
    cbar = plt.colorbar()
    phase_items = sorted(phaseslabels.items(), key=lambda x:x[1])
    labels = [item[0] for item in phase_items]
    values = [item[1] for item in phase_items]
    
    cbar.set_ticks(values)
    cbar.set_ticklabels(labels)
    
    
    plt.show()
        
    return None

"""
mineral group functions
"""

def def_minerals(path, csv):
    """
    Create dictionary of inserted mineral phases 

    Parameters
    ----------
    path : TYPE, str
        DESCRIPTION: path to csv file 
    csv : TYPE, str
        DESCRIPTION: file name of csv

    Returns
    -------
    minerals : dict(mineralphase name: minerlphase class)
        DESCRIPTION.

    """
    #open csv with mineral phase data
    minerals = {}
    data = pd.read_csv(path+csv)
    
    #create mineralphase class for each mineral phase in csv doc
    for i, phase in enumerate(data['Name']):
        minerals[phase] = mineral_phase(phase,float(data['Density'].iloc[i]), data['Major Compositions'].iloc[i])
    
    return minerals


def composition(path, csv, mineral_phase):
    """
    Add composition of each element to each inserted mineral phase

    Parameters
    ----------
    path : TYPE, str
        DESCRIPTION: path to csv file
    csv : TYPE, str
        DESCRIPTION: filename of csv doc
    mineral_phase : TYPE, dict(mineralphase name: mineralphase class)
        DESCRIPTION: dictionary of mineralphases inserted

    Returns
    -------
    None.

    """
    
    
    data = pd.read_csv(path+csv).set_index("Name")
    Z = np.arange(start=3, stop=93)
    elements = {x: xrl.AtomicNumberToSymbol(x) for x in Z} # includes Li to U
    for phase in mineral_phase:
        conc = [data.loc[phase, elements[x]] for x in Z]
        composition = {elements[i]: conc[n] for n, i in enumerate(elements)}
        mineral_phase[phase].add_composition(composition)

    return None



def minphase_cs_width(mineral, energy, err=5):
    """
    Add range of linear attenuation coeffs to each inserted mineralphase
    
    error is percentual

    Parameters
    ----------
    mineral : TYPE, dict(mineralphase name: mineralphase class)
        DESCRIPTION: dictionary of inserted mineral phases
    energy : TYPE, float
        DESCRIPTION: energy of x-ray beam used for xrf/ct measurement
    err : TYPE,float
        DESCRIPTION: error to be taken on linear attenuation coeff, DEFAULT IS 5%

    Returns
    -------
    None.

    """
    for phase in mineral.values():
        composition = phase.major_composition
        
        assert phase.density != 0 or float("Nan"), "Density is impossible (0 or unfilled)"
        
        #calculate linear attenuation coeff range using xraylib
        cs = xrl.CS_Energy_CP(composition, energy)*phase.density 
        cs_min, cs_max = (1-err/100)*cs, (1+err/100)*cs
        
        #add linear atten coeff range to mineral phase object
        phase.add_attenuation_range([cs_min, cs, cs_max])
        
    return None




def create_minerals(path, csv, energy, err=5):
    """
    Create dictionary of mineralphases with attenuation coeff ranges and compositions

    Parameters
    ----------
    path : TYPE, str
        DESCRIPTION: path to where csv file is located
    csv : TYPE, str
        DESCRIPTION: csv file name with mineralphase data
    energy : TYPE, float
        DESCRIPTION: energy of x-ray beam used in xrf/ct measurement
    err : TYPE, float
        DESCRIPTION: percentual error for calculation of linear attenuation coefficient range, DEFAULT is 5%

    Returns
    -------
    minphases : TYPE, dict(mineralphase name: mineralphase class)
        DESCRIPTION: dictionary of mineralphases from csv

    """
    #create mineral class objects and add attenuation range + composition
    minphases = def_minerals(path, csv)
    minphase_cs_width(minphases, energy, err=err)
    composition(path, csv, minphases)
    
    return minphases



def minerals(path_file, filename, path_csv, csv, energy):
    """
    Write mineralphase data to h5 file

    Parameters
    ----------
    path_file : TYPE, str
        DESCRIPTION: path to h5 file to write in/create
    filename : TYPE, str
        DESCRIPTION: file name of h5 to create/ write in
    path_csv : TYPE, str
        DESCRIPTION: path to csv file with mineralphase data
    csv : TYPE, str
        DESCRIPTION: file name of csv with mineralphase data
    energy : TYPE, float
        DESCRIPTION: energy of used x-ray beam in xrf/ct measurement

    Returns
    -------
    None.

    """
    #create mineral phase objects
    minphases = create_minerals(path=path_csv, csv=csv, energy=energy)
    
    datasets = ["Names", "Major Composition", "Density", "Elemental Composition", "Cross-section Range"]
    
    #make list of each class component for clear grouping in h5 file
    names = [x for x in minphases]
    major_compositions = [minphases[x].major_composition for x in minphases]
    compositions = [minphases[x].composition for x in minphases]
    densities = [minphases[x].density for x in minphases]
    cs_range = [minphases[x].atten_range for x in minphases]
    
    #write data to h5 file
    with h5py.File(path_file + filename+".h5", 'w') as f:
        try:
            f.create_group("mineral phases")
            
        except Exception:
            pass
        
        for dataset in datasets:
            if dataset in f["mineral phases"]:
                del f["mineral phases"][dataset]

                
        f['mineral phases'].create_dataset("Names", data = names )
        f["mineral phases"].create_dataset("Major Composition", data= major_compositions)
        f['mineral phases'].create_dataset("Density", data = densities)
        f["mineral phases"].create_dataset("Elemental Composition", data= compositions)
        f['mineral phases'].create_dataset("Cross-section range", data = cs_range)
    
    return None





def def_pixels(pix, seen_min, minerals,  ps_cor): 
    """
    Create dictionary of pixelclass

    Parameters
    ----------
    pix : TYPE, 2d list
        DESCRIPTION: 2d np array of pixel values
    seen_min : TYPE, list
        DESCRIPTION: list of seen mineralphases ordered per pixel index
    ps_cor : TYPE, float
        DESCRIPTION: pixel to real size correlation

    Returns
    -------
    Pixs : TYPE,dict(index, pixels class)
        DESCRIPTION: dictionary of pixelclass with pixel values, mineralphases and correlation to real sizes

    """
    
    #starting index from 0; pix = 2d np array
    Pixs = {}
    for i, val in enumerate(pix):
        Pixs[i] = pixels(i, val, seen_min[i], val.shape, ps_cor)
    
    #add dummy pixel -> necessary for rotation
    dummy_value = -1
    Pixs[dummy_value] = pixels(dummy_value, dummy_value, minerals["Dummy"] , (1,1), ps_cor )
    
    
    return Pixs



def seen_min(pixels, minerals): 
    """
    Create list of mineralphases for each pixel given

    Parameters
    ----------
    pixels : TYPE, 2d list
        DESCRIPTION: 2d np array given by pixelisation function
    minerals : TYPE, dict(mineralphase name: mineralphase class)
        DESCRIPTION: dictionary of mineralphases with inserted mineralphase data

    Returns
    -------
    seen_min : TYPE, list
        DESCRIPTION: list of seen mineralphases based on linear attenuation coeff range

    """
    
    #pixels = 2d np array and minerals = dict minerals class
    seen_min = {}
    for index, pix in enumerate(pixels):
        
        #calculate average value of pixels and see if in attenuation coeff range
        avg = np.mean(pix)
        pos_min = []
        
        if avg == -1: # add dummy mineralphase to use when rotating image and dummy data gets added
            pos_min.append((minerals["Dummy"], 0))
        
        for phase in minerals.values():
            lcoeff, hcoeff = phase.atten_range[0], phase.atten_range[2]
            normal = phase.atten_range[1]
            
            if avg >= lcoeff and avg <= hcoeff:
                pos_min.append((phase, abs(avg - normal)))
        
        if len(pos_min) == 0:
            pos_min.append((minerals["Matrix"], 0))
        
        best = min(pos_min, key= lambda x: x[1])
        best_phase = best[0]
        seen_min[index] = best_phase
        
    return seen_min



def create_pixels(data, mineralphases, path_img=None, image=None, array=[], pixel_size_y=2, pixel_size_x=2,path_min='', csv='', energy=21, ps_cor=1, err=5):
    """
    Create dict(index: pixels class) with adjusted ct scans 

    Parameters
    ----------
    data : TYPE, nd array
        DESCRIPTION: nd array of grayscaled datapoints for each phase
    mineralphases : TYPE, list
        DESCRIPTION: names of mineralphases, corresponding to data
    path_img : TYPE, str
        DESCRIPTION: path to image file location
    image : TYPE, str
        DESCRIPTION: image file name
    array : TYPE, 2d list
        DESCRIPTION: optional parameter for code tests, DEFAULT IS list()
    pixel_size_y : TYPE, int
        DESCRIPTION: amount of rows the pixels should be in size, DEFAULT is 2
    pixel_size_x : TYPE, int
        DESCRIPTION: amount of columns the pixels should be in size, DEFAULT is 2
    path_min : TYPE, str
        DESCRIPTION: path to csv file with mineralphase data
    csv : TYPE, str
        DESCRIPTION: file name of csv with mineralphase data
    energy : TYPE, float
        DESCRIPTION: energy of used x-ray beam in xrf/ct measurement, DEFAULT is 21 keV
    ps_cor : TYPE, float
        DESCRIPTION: pixel to real size correlation coefficient, DEFAULT is 1 pixel: 1 µm
    err: TYPE, float
        DESCRIPTION: percentual error to calculate linear attenuation coefficient ranges, DEFAULT is 5%

    Returns
    -------
    pixs : TYPE, pixelclass
        DESCRIPTION: dictionary with indexes of pixels and corresponding pixelclass

    """
    
    #energy in keV
    print("Initialising mineral phases...", end='')
    mineral_phases = create_minerals(path_min, csv, energy, err)    
    
    print('Done')
    print("Rescaling image values...", end='')
    
    #using lin regress turn image grayscale values into lin atten coeff range
    img = ct_scan(path_img, image) if path_img and image is not None else array
    slope, intercept = linregress(data, mineralphases,mineral_phases, energy=energy)[:2]
    img_cor = (img - intercept)/ slope
    
    print("Done")
    
    #create pixels
    print("Generating pixels...", end='')
    pixels = pixelisation(array=img_cor, cols=pixel_size_x, rows=pixel_size_y)
 
    #add mineral phases to pixels and create pixel class objects
    
    seen_minerals = seen_min(pixels, mineral_phases)
    pixs = def_pixels(pixels, seen_minerals, mineral_phases, ps_cor)
    print("Done")
    return pixs




#need to create to file if have time -> not fully necessary tbh
#def create_pixels_file(path_img= '', imagefile='', array=[], rows=2, cols=2,mineralphasepath='', mineralphasefile='', energy=21, ps=1):
    

def display_histogram(img, bar=False, scale='log'):
    """
    plots a barplot or intensity chart of image intensity values

    Parameters
    ----------
    img : TYPE, 2d list
        DESCRIPTION: 2d array of original image
    bar : TYPE, boolean
        DESCRIPTION: describe if barplot or intensity plot should be given,
        DEFAULT is False
        barplots significantly slow down calculations
    scale: TYPE, str
        DESCRIPTION: scaling of y axis, DEFAULT is logarithmic

    Returns
    -------
    None.

    """

    #flatten image to 1d array
    flat_img = img.ravel()
    
    #calculate histogram frequencies with correct amount of bins
    minval = np.min(flat_img)
    maxval = np.max(flat_img)
    hist = np.bincount(flat_img)[minval:maxval+1]

    
    #plot histogram
    if bar is False:
        plt.plot(np.arange(minval, maxval+1), hist)
        plt.yscale(scale)
        plt.ylabel('frequency')
        plt.xlabel('pixel value')
        plt.show()
        
    else:
        plt.bar(np.arange(minval, maxval+1), hist)
        plt.yscale(scale)
        plt.ylabel('frequency')
        plt.xlabel('pixel value')
        plt.show()
    
    return None


def hist_to_phasedata(img, nphases, sigmas,width=1000, window=3):
    """
    find the peaks in a smoothened + filtered histogram plotted via display_histogram
    

    Parameters
    ----------
    img : TYPE, 2d list
        DESCRIPTION: image of multiple mineral phases
    nphases : TYPE, int
        DESCRIPTION: amount of phases that can be seen in image
    sigmas : TYPE, int or sequence
        DESCRIPTION: estimates of std of gaussian distributions in histogram
    width : TYPE, int or sequence
        DESCRIPTION: width for scipy wavelet peak finding function, DEFAULT is 1000.
    window : TYPE, int (odd)
        DESCRIPTION: size of window for median filtereing, DEFAULT is 3.

    Returns
    -------
    phases_peaks : TYPE, list
        DESCRIPTION: peak intensity average for each mineral phase

    """
    
    #flatten image to work with bincount function
    flat_img = img.ravel()
    minval, maxval = np.min(flat_img), np.max(flat_img)
    
    
    #calculate rough frequency
    frequency = np.bincount(flat_img)[minval:maxval+1]
    
    
    
    #remove spikes using median filter
    filtered_freq = sc.signal.medfilt(frequency, window)
    
    #filter/smoothe frequencies and  calculate the relative maxima
    refreq = sc.ndimage.gaussian_filter1d(filtered_freq, sigmas)
    peaks = sc.signal.find_peaks_cwt(refreq, width)
    print(peaks)
    
    #search for largest peaks
    intensities = {refreq[x]: x for x in peaks}
    largest_peaks = sorted(refreq[peaks], reverse=True)
    
    phases_peaks = [intensities[largest_peaks[x]] for x in range(nphases)]
    
    return phases_peaks




def linregress(data, mineralphases_present, mineralphases, energy=21):
    """
    perform linear regression, based on background and standard known phases in tif/tiff image

    Parameters
    ----------
    data : TYPE, list
        DESCRIPTION: list of dataopoints from grayscaled image
    mineralphases_present : TYPE, list
        DESCRIPTION: list of mineralphases that represent the datapoints
    mineralphases : TYPE, dictionary(mineralphase class)
        DESCRIPTION: dictionary of mineralphases that have been added to csv file
    energy : TYPE, float
        DESCRIPTION: energy of used x-ray beam in xrf/ct measurement, DEFAULT is 21 keV

    Returns
    -------
    slope : TYPE, float
        DESCRIPTION: slope of the regression line
    intercept : TYPE, float
        DESCRIPTION: intercept of the regression line
    r : TYPE, float
        DESCRIPTION: the Pearson correlation coefficient
    p : TYPE, float
        DESCRIPTION: the p-value for a hypothesis test whose null hypothesis is that the slope is zero 
        using Wald Test with t-distribution
    sterr : TYPE, float
        DESCRIPTION: standard error of the estimated intercept, under assumption of residual normality

    """
    mus = [xrl.CS_Energy_CP(mineralphases[x].major_composition, energy)*mineralphases[x].density for x in mineralphases_present]
    slope, intercept, r, p, sterr = sc.stats.linregress(mus, data)
    
    return slope, intercept, r, p, sterr

        
def nlayers_x(Pixels, traversed_pixelnr, image_size, pixel_size):
    """
    calculate the pixels before the traversed pixels

    Parameters
    ----------
    Pixels : TYPE, pixelclass
        DESCRIPTION: pixels created through create_pixels function
    traversed_pixelnr : TYPE, int
        DESCRIPTION: index of traversed pixel
    image_size : TYPE, tuple
        DESCRIPTION: tuple with size of original image (rows, columns)
    pixel_size : TYPE, tuple
        DESCRIPTION: tuple with size of pixels (rows, columns)

    Returns
    -------
    TYPE, array
        DESCRIPTION: array with pixels that come before inputted pixel index per row

    """
    #calculate how large each row is
    traversed_x = []
    row_length = image_size[1] // pixel_size[1]
    assert image_size[1] % pixel_size[1] == 0, "correct pixelisation did not occur"
    
    #calculate in which row the traversed pixel is & add those before traversed pixel
    rownr = traversed_pixelnr // row_length
    for n, i in enumerate(Pixels.values()):
        if n in np.arange(rownr * row_length, traversed_pixelnr, 1):
            traversed_x.append(i)
    
    return np.array(traversed_x)
        

def nlayers_y(Pixels, traversed_pixelnr, image_size, pixel_size):
    """
    calculate the pixels above the traversed pixel

    Parameters
    ----------
    Pixels : TYPE, pixelclass
        DESCRIPTION: pixels created through create_pixels function
    traversed_pixelnr : TYPE, int
        DESCRIPTION: index of traversed pixel
    image_size : TYPE, tuple
        DESCRIPTION: tuple with size of original image (rows, columns)
    pixel_size : TYPE, tuple
        DESCRIPTION: tuple with size of pixels (rows, columns)

    Returns
    -------
    TYPE, array
        DESCRIPTION: array with pixels that are directly above the inputted pixel

    """
    row_length = image_size[1] // pixel_size[1]
    assert image_size[1] % pixel_size[1] == 0, "correct pixelisation did not occur"
    
    rownr = traversed_pixelnr // row_length
    column = traversed_pixelnr % row_length # index in chosen row
    n = [r*row_length + column for r in range(rownr+1)] # indexes of all pixels above chosen pixel

    traversed = [Pixels[i] for i in n]
    
    return np.array(traversed)


def rows_pixels(Pixels, image_size):
    """
    gives the pixels per row

    Parameters
    ----------
    Pixels : TYPE, pixelclass
        DESCRIPTION: pixels created through create_pixels function
    image_size : TYPE, tuple
        DESCRIPTION: tuple with size of original image (rows, columns)
    pixel_size : TYPE, tuple: optional
        DESCRIPTION: tuple with size of original image, DEFAULT is pixel[0].shape

    Returns
    -------
    rows : TYPE, 2d array
        DESCRIPTION: 2d array of pixels 

    """
    indexmatrix = rows_index(Pixels, image_size)
    pixelmatrix = []
    for row in indexmatrix:
        row = [Pixels[index] for index in row]
        pixelmatrix.append(row)
    
    return np.array(pixelmatrix)


#can be made using more elegant code, but works
def rows_index(Pixels, image_size, pixel_size=None):
    """
    gives the pixels per row

    Parameters
    ----------
    Pixels : TYPE, pixelclass
        DESCRIPTION: pixels created through create_pixels function
    image_size : TYPE, tuple
        DESCRIPTION: tuple with size of original image (rows, columns)
    pixel_size : TYPE, tuple: optional
        DESCRIPTION: tuple with size of original image, DEFAULT is pixel[0].shape

    Returns
    -------
    rows : TYPE, 2d array
        DESCRIPTION: 2d array of pixel indexes

    """
    #calculate length of each row
    pixel_size = Pixels[0].shape if pixel_size is None else pixel_size
    
    row_length = image_size[1] // pixel_size[1]
    assert image_size[1] % pixel_size[1] == 0, "correct pixelisation did not occur"
    
    #append each pixel to a list before adding to rows list
    rows = []
    row, index = [], 0
    for p in list(Pixels.keys())[:-1]:
        row.append(p)
        index += 1
        
        if index == row_length:
            rows.append(row)
            row = []
            index = 0
    
    #pad with dummy pixels to rotate later in sinogram
    padded_rows = pad_img(np.array(rows))
    print(f'---------- \nenlarged image shape is {padded_rows.shape}\n----------')
    
    return padded_rows


#recheck cumprod
def absorption_x(Pixels, energy, image_size, pixel_size=None):
    """
    calculate an absoprtion matrix in the horizontal direction (entry)

    Parameters
    ----------
    Pixels : TYPE, Pixelclass
        DESCRIPTION: Pixels created through create_pixel function
    energy : TYPE, float
        DESCRIPTION: energy used for xrf/ct measurement
    image_size : TYPE, tuple
        DESCRIPTION: tuple with image size (rows, columns)
    pixel_size : TYPE, tuple
        DESCRIPTION: tuple with pixel size (rows, columns)

    Returns
    -------
    absorp_total : TYPE, 2d array
        DESCRIPTION: 2d array of calculated entry absorption

    """
    rearranged_indexes = rows_index(Pixels, image_size, pixel_size)
    absorp_total = []
    
    for row in rearranged_indexes:
        absorp = []
        ab_b = 1
        
        for pixelindex in row:
            density = Pixels[pixelindex].phase.density
            thickness = Pixels[pixelindex].cor
            mu = xrl.CS_Energy_CP(Pixels[pixelindex].phase.major_composition, energy)
            
            ab_c = np.exp(-1*mu*thickness*density)
            ab = np.prod([ab_b, ab_c])
            absorp.append(ab)
            ab_b = ab_c
            
        absorp_total.append(absorp)
    
    return np.array(absorp_total)

#recheck cumprod
def absorption_y(Pixels, element, line, image_size, pixel_size=None):
    """
    calculate an absoprtion matrix in the vertical direction (exit)

    Parameters
    ----------
    Pixels : TYPE, Pixelclass
        DESCRIPTION: Pixels created through create_pixel function 
    element : TYPE, str
        DESCRIPTION: element that is chosen to be analysed
    line : TYPE, int
        DESCRIPTION: line macro of xraylib
    image_size : TYPE, tuple
        DESCRIPTION: tuple with image size (rows, columns)
    pixel_size : TYPE, tuple
            DESCRIPTION: tuple with pixel size (rows, columns)

        Returns
        -------
        absorp_total : TYPE, 2d array
            DESCRIPTION: 2d array of calculated entry absorption

    """
    rearranged_indexes = rows_index(Pixels, image_size, pixel_size)
    absorp_total = []
    ab_b = np.ones(len(rearranged_indexes[0]))
    for row in rearranged_indexes:
        absorp = []
        
        ab_cs = []
        for n, pixelindex in enumerate(row):
            density = Pixels[pixelindex].phase.density
            thickness = Pixels[pixelindex].cor
            Z = xrl.SymbolToAtomicNumber(element)
            energy = xrl.LineEnergy(Z, line)
            mu = xrl.CS_Energy_CP(Pixels[pixelindex].phase.major_composition, energy)
            
            ab_c = np.exp(-1*mu*thickness*density)
            ab_cs.append(ab_c)
            
            ab = np.prod([ab_b[n], ab_c])
            absorp.append(ab)
        
        absorp_total.append(absorp)
        ab_b = ab_cs
            
    return np.array(absorp_total)
    



def fluorescence(Pixels, energy, image_size, pixel_size=None):
    """
    calculate the fluorescence matrix

    Parameters
    ----------
    Pixels : TYPE, Pixelclass
        DESCRIPTION: Pixels created through create_pixel function
    energy : TYPE, float
        DESCRIPTION: energy used for xrf/ct measurement
    image_size : TYPE, tuple
        DESCRIPTION: tuple with image size (rows, columns)
    pixel_size : TYPE, tuple
        DESCRIPTION: tuple with pixel size (rows, columns)

    Returns
    -------
    absorp_total : TYPE, 2d array
        DESCRIPTION: 2d array of calculated fluorescence


    """
    rearranged_indexes = rows_index(Pixels, image_size, pixel_size)
    fluor_total = []
    
    for row in rearranged_indexes:
        fluo = []
        
        for pixelindex in row:
            density = Pixels[pixelindex].phase.density
            thickness = Pixels[pixelindex].cor
            mu = xrl.CS_Energy_CP(Pixels[pixelindex].phase.major_composition, energy)
            
            ab = 1- np.exp(-1*mu*thickness*density)
            fluo.append(ab)
            
        fluor_total.append(fluo)
    
    return np.array(fluor_total)
    

def production_cross_section(element, line, energy):
    """
    use xraylib to calculate the xrf production cross-section

    Parameters
    ----------
    element : TYPE, str
        DESCRIPTION: symbol of element
    line : TYPE, int
        DESCRIPTION: xrf line macro in xraylib
    energy : TYPE, float
        DESCRIPTION: energy of XRF measurement

    Returns
    -------
    prod_cs : TYPE, float
        DESCRIPTION: production cross-section of given element, line and energy

    """
    Z = xrl.SymbolToAtomicNumber(element)
    prod_cs = xrl.CS_FluorLine(Z, line, energy)
    
    return prod_cs


#ask laszlo for changes in a and b
def Geometryfactor(Adet, Dsd):
    """
    calculate the Geometry factor for the fpm formula:
        G = Adet / (4 * pi * sin a) with a = 90°

    Parameters
    ----------
    Adet : TYPE, float
        DESCRIPTION: active area of detector in cm^2
    Dsd : TYPE, float
        DESCRIPTION: distance from sample to detector in cm

    Returns
    -------
    G : TYPE, float
        DESCRIPTION: geometry factor for xrf setup (a=90°)

    """
    omega_det = Adet / Dsd**2
    G = omega_det / (4*np.pi)
    
    return G
    


def w_mu(Pixels, element, energy, image_size, pixel_size=None):
    """
    calculate the weight fraction matrix for a given image

    Parameters
    ----------
    Pixels : TYPE, pixelclass
        DESCRIPTION: pixels created through create_pixels function
    element : TYPE, str
        DESCRIPTION: string of elemental symbol
    energy  : TYPE, float
        DESCRIPTION: energy of XRF measurement
    image_size : TYPE, tuple
        DESCRIPTION: tuple of image size (rows, columns)
    pixel_size : TYPE, tuple
        DESCRIPTION: tuple of pixel size (rows, columns), optional

    Returns
    -------
    wi : TYPE, 2d array
        DESCRIPTION: matrix for weight fractions over mu of pixels

    """
    rearranged_indexes = rows_index(Pixels, image_size, pixel_size=pixel_size)
    wi = []
    
    for row in rearranged_indexes:
        w = []
        
        for pixelindex in row:
            w_pix = ppm_to_weightfrac(Pixels[pixelindex].conc[element])
            mu_o = mu_o = xrl.CS_Energy_CP(Pixels[pixelindex].phase.major_composition, energy)
            w_mu_pix = w_pix / mu_o
            
            w.append(w_mu_pix)
                                      
            
        wi.append(w)
        
    return np.array(wi)
    


def fundamental_parameters(Adet, Dsd, element, line, energy, Pixels, image_size, pixel_size=None):
    """
    calculate and return the fundamental parameters necessary for intensity calculations

    Parameters
    ----------
    Adet : TYPE, float
        DESCRIPTION: active area of detector in cm^2
    Dsd : TYPE, float
        DESCRIPTION: distance from sample to detecor in cm
    element : TYPE, str
        DESCRIPTION: string of elemental symbol
    line : TYPE, int
        DESCRIPTION: macro line from xraylib
    energy : TYPE, float
        DESCRIPTION: energy of xray beam used for xrf/ct measurement
    Pixels : TYPE, pixelclass
        DESCRIPTION: pixels created through create_pixels function
    image_size : TYPE, tuple
        DESCRIPTION: tuple of image size (rows, columns)
    pixel_size : TYPE, tuple
        DESCRIPTION: tuple of pixel size, optional

    Returns
    -------
    G : TYPE, float
        DESCRIPTION: geometry factor for xrf setup (a=90°)
    Qi : TYPE, float
        DESCRIPTION: production cross section for elemental xrf line
    w_mu : TYPE, 2d array
        DESCRIPTION: matrix for weight fractions over mu of pixels
    fluor_total : TYPE, 2d array
        DESCRIPTION: matrix for fluorescence factor
    absorption_x_total : TYPE, 2d array
        DESCRIPTION: matrix for entry absorption (x)
    absorption_y_total : TYPE, 2d array
        DESCRIPTION: matrix for exit absorption (y)

    """
    #calculate non matrix parameters first
    G = Geometryfactor(Adet, Dsd)
    Qi = production_cross_section(element, line, energy)
    
    #initialise matrix for each leftover parameter
    w_mu = []
    fluor_total = []
    absorption_x_total = []
    absorption_y_total = []
    
    #create a matrix of the pixel index numbers
    rearranged_indexes = rows_index(Pixels, image_size, pixel_size=pixel_size)
    
    #initialise necessary absorption in exit
    ab_b_y = np.ones(len(rearranged_indexes[0])) # create array of row length
 
    
    for row in rearranged_indexes:
        #calculate the weight concentration for a certain element per pixel/row
        w = []
        
        
        #absorption for entry(x)
        absorp_x = []
        ab_b_x = 1
        
        #absorption for exit (y)
        absorp_y = []
        ab_cs_y = []
        
        #fluorescence contribution per row
        fluor = []
        
        
        for n, pixelindex in enumerate(row):
            
            #calculate necessary parameters to calculate 3 interactions
            density = Pixels[pixelindex].phase.density
            thickness = Pixels[pixelindex].cor
            mu_o = xrl.CS_Energy_CP(Pixels[pixelindex].phase.major_composition, energy)
            
            Z = xrl.SymbolToAtomicNumber(element)
            energy_xrf = xrl.LineEnergy(Z, line) #
            mu_f = xrl.CS_Energy_CP(Pixels[pixelindex].phase.major_composition, energy_xrf)
            
            
            #calculate weight fraction over mu per pixel
            w_pix = ppm_to_weightfrac(Pixels[pixelindex].conc[element]) / mu_o
            w.append(w_pix)
            
            #calculate absorption in entry (x) axis
            ab_c = np.exp(-1*mu_o*thickness*density)
            ab = np.prod([ab_b_x, ab_c])
            absorp_x.append(ab)
            ab_b_x = ab_c
            
            
            #calculate absorption in exit (y) axis
            ab_c_y = np.exp(-1*mu_f*thickness*density)
            ab_cs_y.append(ab_c_y)
            
            ab_y = np.prod([ab_b_y[n], ab_c_y])
            absorp_y.append(ab_y)
            
        
            #calculate fluorescence contribution
            contr_fluo = 1- np.exp(-1*mu_o*thickness*density)
            fluor.append(contr_fluo)
        
        w_mu.append(w)    
        
        absorption_x_total.append(absorp_x)
        
        absorption_y_total.append(absorp_y)
        ab_b_y = ab_cs_y
        
        fluor_total.append(fluor)
        
        
    return G, Qi, np.array(w_mu), np.array(fluor_total), np.array(absorption_x_total), np.array(absorption_y_total)


def FP(density, thickness, concs, mu_o, mu_f):
    """
    calculate and return the fundamental parameters necessary for intensity calculations, using already made indexmatrix

    Parameters
    ----------
    density : TYPE, 2d array
        DESCRIPTION: 2d array of density of mineral phase per pixel
    thickness : TYPE, 2d array
        DESCRIPTION: 2d array of thickness per pixel
    concs : TYPE, 2d array
        DESCRIPTION: 2d array of elemental concentration per pixel
    mu_o : TYPE, 2d array
        DESCRIPTION: 2d array of linear attenuation coeffs at original x-ray beam energy per pixel
    mu_f : TYPE, 2d array
        DESCRIPTIONL: 2d array of linear attenuation coeffs at fluorescence energy per pixel
    ----------
    w_mu : TYPE, 2d array
        DESCRIPTION: matrix for weight fractions over mu of pixels
    fluor_total : TYPE, 2d array
        DESCRIPTION: matrix for fluorescence factor
    absorption_x_total : TYPE, 2d array
        DESCRIPTION: matrix for entry absorption (x)
    absorption_y_total : TYPE, 2d array
        DESCRIPTION: matrix for exit absorption (y)

    """
    
    
    # Calculate fundamental parameters
    w_mu = concs / mu_o
    fluor_total = 1 - np.exp(-mu_o * thickness * density)
    
    # Compute cumulative absorption along X and Y
    
    absorption_x_total = np.cumprod(
        np.exp(-mu_o * thickness * density), axis=1
    )
    
    
    absorption_y_total = np.cumprod(
        np.exp(-mu_f * thickness * density), axis=0
    )
    
    
    return w_mu, fluor_total, absorption_x_total, absorption_y_total



def preprocess_param(Pixels, indexmatrix, element, line, energy):
    """
    process necessary parameters for FPM formula

    Parameters
    ----------
    Pixels : TYPE, pixelclass
        DESCRIPTION: dictionary of pixels generated through create_pixels function
    indexmatrix : TYPE, 2d array
        DESCRIPTION: matrix of pixelindices
    element : TYPE, str
        DESCRIPTION: string representation of elemental symbol
    line : TYPE, int
        DESCRIPTION: macro of line in xraylib
    energy : TYPE, float
        DESCRIPTION: energy of used x-ray beam

    Returns
    -------
    density : TYPE, array
        DESCRIPTION: matrix of densities per pixel
    thickness : TYPE, array
        DESCRIPTION: matrix of thickness per pixel (cte)
    concentration : TYPE, array
        DESCRIPTION: matrix of concentration of specified element per pixel
    mu_o : TYPE, array
        DESCRIPTION: matrix of linear attenuation coeffs at original x-ray beam energy per pixel
    mu_f : TYPE, array
        DESCRIPTION: matrix of linear attenuation coeffs at xrf energy per pixel
            

    """
    density, thickness, concentration, mu_o, mu_f =[], [], [], [], [] 
    
    
    for row in indexmatrix:
        density_row = []
        thickness_row = []
        conc_row = []
        mu_o_row = []
        mu_f_row = []
        
        Z = xrl.SymbolToAtomicNumber(element) 
        energy_xrf = xrl.LineEnergy(Z, line) 
        
        for index in row:
            rho = Pixels[index].phase.density
            T = Pixels[index].cor
            conc = ppm_to_weightfrac(Pixels[index].conc[element])
            mu_o_pix = xrl.CS_Energy_CP(Pixels[index].phase.major_composition, energy)
            mu_f_pix = xrl.CS_Energy_CP(Pixels[index].phase.major_composition, energy_xrf)
            
            
            
            density_row.append(rho)
            thickness_row.append(T)
            conc_row.append(conc)
            mu_o_row.append(mu_o_pix)
            mu_f_row.append(mu_f_pix)
        
        density.append(density_row)
        thickness.append(thickness_row)
        concentration.append(conc_row)
        mu_o.append(mu_o_row)
        mu_f.append(mu_f_row)
    
    return np.array(density), np.array(thickness), np.array(concentration), np.array(mu_o), np.array(mu_f)
        
    
    
def FP_old(Pixels, indexmatrix, element, line, energy): 
     #initialise matrix for each parameter 
     w_mu = [] 
     fluor_total = [] 
     absorption_x_total = [] 
     absorption_y_total = [] 
     
     #initialise necessary absorption in exit 
     ab_b_y = np.ones(len(indexmatrix[0])) # create array of row length 
     
     for row in indexmatrix: 
         #calculate the weight concentration for a certain element per pixel/row 
         w = [] 
         
         #absorption for entry(x) 
         absorp_x = [] 
         ab_b_x = 1 
         
         #absorption for exit (y) 
         absorp_y = []
         ab_cs_y = [] 
         #fluorescence contribution per row 
         fluor = [] 
         
         for n, pixelindex in enumerate(row): 
             #calculate necessary parameters to calculate 3 interactions 
             density = Pixels[pixelindex].phase.density 
             thickness = Pixels[pixelindex].cor 
             mu_o = xrl.CS_Energy_CP(Pixels[pixelindex].phase.major_composition, energy) 
             
             Z = xrl.SymbolToAtomicNumber(element) 
             energy_xrf = xrl.LineEnergy(Z, line) 
             mu_f = xrl.CS_Energy_CP(Pixels[pixelindex].phase.major_composition, energy_xrf)
             
             #calculate weight fraction over mu per pixel 
             w_pix = ppm_to_weightfrac(Pixels[pixelindex].conc[element]) / mu_o 
             w.append(w_pix)
             
             #calculate absorption in entry (x) axis 
             ab_c = np.exp(-1*mu_o*thickness*density) 
             ab = np.prod([ab_b_x, ab_c]) 
             absorp_x.append(ab) 
             ab_b_x = ab_c 
             
             #calculate absorption in exit (y) axis 
             ab_c_y = np.exp(-1*mu_f*thickness*density)
             ab_cs_y.append(ab_c_y) 
             
             ab_y = np.prod([ab_b_y[n], ab_c_y])
             absorp_y.append(ab_y) 
             
             #calculate fluorescence contribution 
             contr_fluo = 1- np.exp(-1*mu_o*thickness*density)
             fluor.append(contr_fluo)
         w_mu.append(w)
         
         
         absorption_x_total.append(absorp_x)
         absorption_y_total.append(absorp_y)
         ab_b_y = ab_cs_y 
         fluor_total.append(fluor)
             
     return np.array(w_mu), np.array(fluor_total), np.array(absorption_x_total), np.array(absorption_y_total)
    
#LOOK AT THIS ONE !!!!
def I_fluor(I0, G, Qi, w_mu, fluormatrix, absorption_x, absorption_y):
    """
    calculate theorhetical intensity of specific xrf line for chosen element,
        uses fpm formula derived from tom schoonjans,
        fpm parameters already calculated


    Parameters
    ----------
    I0 : TYPE, float
        DESCRIPTION: original xraybeam intensity
    G : TYPE, float
        DESCRIPTION: geometry factor calculate through geometryfactor function, 
        geometry factor for xrf setup (a=90°)
    Qi : TYPE, float
        DESCRIPTION: production cross sections calculated through prod_cross_section function
    w_mu : TYPE, 2d array
        DESCRIPTION: matrix for weight fractions over mu of pixels
    fluormatrix : TYPE, 2d array
        DESCRIPTION: matrix for fluorescence factor
    absorption_x : TYPE, 2d array
        DESCRIPTION: matrix for entry absorption (x)
    absorption_y : TYPE, 2d array
        DESCRIPTION: matrix for exit absorption (y)

    Returns
    -------
    intensity : TYPE, 1d array
        DESCRIPTION: 1d array of calculated theorhetical intensities for each row (entry point)


    """
    #calculate intensity
    matrix = w_mu*absorption_x*fluormatrix*absorption_y
    smatrix = np.sum(matrix, axis=1)
    intensity = I0*G*Qi*smatrix
    
    return intensity
    


def Intensity_i(I0, Adet, Dsd, element, line, energy, Pixels, image_size, pixel_size=None): 
    """
    calculate theorhetical intensity of specific xrf line for chosen element,
        uses fpm formula derived from tom schoonjans

    Parameters
    ----------
    I0 : TYPE, float
        DESCRIPTION: original xraybeam intensity
    Adet : TYPE, float
        DESCRIPTION: active area of detector in cm^2
    Dsd : TYPE, float
        DESCRIPTION: distance from sample to detecor in cm
    element : TYPE, str
        DESCRIPTION: string of elemental symbol
    line : TYPE, int
        DESCRIPTION: macro line from xraylib
    energy : TYPE, float
        DESCRIPTION: energy of xray beam used for xrf/ct measurement
    Pixels : TYPE, pixelclass
        DESCRIPTION: pixels created through create_pixels function
    image_size : TYPE, tuple
        DESCRIPTION: tuple of image size (rows, columns)
    pixel_size : TYPE, tuple
        DESCRIPTION: tuple of pixel size, DEFAULT is None

    Returns
    -------
    intensity : TYPE, 1d array
mor        DESCRIPTION: 1d array of calculated theorhetical intensities for each row (entry point)

    """
    G, Qi, w_over_mu, fluor_total, absorption_x_total, absorption_y_total = fundamental_parameters(Adet, Dsd, element, line, energy, Pixels, image_size, pixel_size=pixel_size)
    matrix = w_over_mu*absorption_x_total*fluor_total*absorption_y_total
    smatrix = np.sum(matrix, axis=1)
    intensity = I0*G*Qi*smatrix
    
    return intensity


def thetas(dtheta, dualdet=False):
    """
    Calculates theta angles in radians from degrees

    Parameters
    ----------
    angles : TYPE, float
        DESCRIPTION: angle of increments

    Returns
    -------
    theta : TYPE, array
        DESCRIPTION: array of uniformly distributed projection angles in radians

    """
    if dualdet is False:
        sections = int(np.ceil(360 / dtheta)+1)
    
    else:
        sections = int(np.ceil(180 / dtheta)+1)
    
    
    theta = tomopy.angles(sections, 0, 360)
    return theta



def sinogram_rotation(I0, Adet, Dsd, element, line, energy, Pixels, image_size, theta, dualdet=False):
    """
    Create a sinogram of XRF intensities calculated using FPM method

    Parameters
    ----------
    I0 : TYPE, float
        DESCRIPTION: Original intensity of x-ray beam in ph/s
    Adet : TYPE, float
        DESCRIPTION: Active detector area in cm^2
    Dsd : TYPE, float
        DESCRIPTION: Distance from sample to detector in cm
    element : TYPE, str
        DESCRIPTION: String of elemental symbol
    line : TYPE, int
        DESCRIPTION: Macro for chosen line in xrl
    energy : TYPE, float
        DESCRIPTION: Energy of x-ray beam used in the experiment
    Pixels : TYPE, Pixel class
        DESCRIPTION: Pixels created through create_pixels function
    image_size : TYPE, tuple(int)
        DESCRIPTION: Tuple of integers, representing the image size
    theta : TYPE, list
        DESCRIPTION: List of angles in radians
    dualdet :  TYPE, boolean    
        DESCRIPTION: True if dual detector setup is used (180°) else 360°, DEFAULT is False 
            
    Returns
    -------
    None

    """
    bt = time.time()
    
    
    #create sinogram
    
    image_indexes = rows_index(Pixels, image_size)
    sinogram_shape = image_indexes.shape[1], len(theta)
    image_shape = image_indexes.shape
    sinogram  =np.zeros(sinogram_shape)
    mask = np.ones(image_shape)
    #calculate intensity for each angle in theta
    
    # FP parameters that dont change with rotation angle of sample
    G = Geometryfactor(Adet, Dsd)
    Qi = production_cross_section(element, line, energy)

    
    #calculate parameters necessary for attenuation/fluorescence matrices
    print("Calculating parameters...", end='')
    b = time.time()
    density, thickness, concentration, mu_o, mu_f = preprocess_param(
                                                Pixels, image_indexes, element, line, energy)
    e = time.time()
    print(f"Done in {e-b} seconds")
    
    
    for n, angle in enumerate(theta):
        angle_degree = angle * (180 / np.pi)
        print(f'Calculating intensity for rotation angle: {angle_degree:.2f}°')
        
        #rotate sample + mask to find distinguish non-dummy data from dummy data
        rotated = sc.ndimage.rotate(image_indexes, angle_degree, reshape=False, order=0)
        rotated_mask = sc.ndimage.rotate(mask, angle_degree, reshape=False, order=0)
        
        rotated[rotated_mask != 1] = -1 #set to dummy value (standard = -1)
        
        rot_dens = sc.ndimage.rotate(density, angle_degree, reshape=False, order=0, cval = 0)
        rot_conc = sc.ndimage.rotate(concentration, angle_degree, reshape=False, order=0, cval=0)
        rot_mu_o = sc.ndimage.rotate(mu_o, angle_degree, reshape=False, order=0, cval=1)
        rot_mu_f = sc.ndimage.rotate(mu_f, angle_degree, reshape=False, order=0, cval=1)
        
        
        #calculate remaining fundamental parameters for this rotated image
        w_mu, fluor, absorption_x, absorption_y = FP(rot_dens, thickness, rot_conc, rot_mu_o, rot_mu_f)
        
        
        #calculate intensity of fluorescence
        intensity = I_fluor(I0, G, Qi, w_mu, fluor, absorption_x, absorption_y)
        
        #append to sinogram
        sinogram[:,n ] = intensity
        
    
    et = time.time()
    print(f"Time taken for calculation sinogram: {et-bt}")
    return np.array(sinogram)





def reconstruction_sinogram(sinogram, theta, algo='fbp'):
    """
    reconstruct image from sinogram

    Parameters
    ----------
    sinogram : TYPE, 2d array
        DESCRIPTION: Sinogram of image
    theta : TYPE, array
        DESCRIPTION: Array of angles in radian
    algo : TYPE, str
        DESCRIPTION: Algorithm to be used in reconstruction, use supported tomopy algorithms, DEFAULT is gridrec
        

    Returns
    -------
    recon : TYPE, 3d array
        DESCRIPTION: Reconstructed image

    """
    #need to make sinogram 3D for tomopy function
    sino3d = sinogram[np.newaxis, :, :]
    print(f"Starting reconstruction using {algo}...", end='')
    #let tomopy perform reconstruction using specified algorithm
    recon = tomopy.recon(sino3d, theta, algorithm=algo, sinogram_order=True)
    print("Done")
    return recon



"""""
final comparison with CI or CM chondrites 

- firstly add CI or CM compositions to .h5 file via excel or text files
   - grouped into its own section "used comparison composition"
- use comparison composition to compare calculated iterative concentrations

"""""



"""
oxide to ppm
"""


def ppm_to_weightfrac(ppm):
    """
    convert ppm concentrations to weightfractions

    Parameters
    ----------
    ppm : TYPE, float
        DESCRIPTION: concentration of element in ppm

    Returns
    -------
    weightfrac : TYPE, float
        DESCRIPTION: concentration of element in weight fraction

    """
    
    weightfrac = ppm / 10000
    
    return weightfrac

def weightfrac_to_ppm(weightfrac):
    """
    convert weightfractions to ppm concentrations

    Parameters
    ----------
    weightfrac : TYPE, float
        DESCRIPTION: concentration of element in weight fraction

    Returns
    -------
    TYPE, float
        DESCRIPTION: concentration of element in ppm

    """
    
    return 1e4*weightfrac

def oxide_ppm_conv(massper, oxide):
    """
    Convert Oxide mass fraction to ppm concentrations

    Parameters
    ----------
    massper : TYPE, float
        DESCRIPTION: oxide masspercentage 
    oxide : TYPE, str 
        DESCRIPTION: oxide

    Returns
    -------
    metal : TYPE, str
        DESCRIPTION: metal
    ppm : TYPE, float
        DESCRIPTION: converted ppm concentration

    """
    compound = xrl.CompoundParser(oxide)
    
    # get mass fraction of metal in the oxide
    for i in range(compound["nElements"]):
         if compound["Elements"][i] != xrl.SymbolToAtomicNumber("O"):
            massfracmetal = compound["massFractions"][i]
            
    #calculate the concentration of m% to ppm -> 10e4
    ppm = int(10**4*massper*massfracmetal)
    metal = xrl.AtomicNumberToSymbol(compound["Elements"][i])
    return metal, ppm


def oxides_ppm_convs(masspercs, oxides):
    """
    Convert multiple oxides to ppm concentrations

    Parameters
    ----------
    masspercs : TYPE, list(floats)
        DESCRIPTION: list of masspercentages of oxides
    oxides : TYPE, list(str)
        DESCRIPTION: list of oxides

    Returns
    -------
    ppms_oxides : TYPE, dictionary(metal: ppm concentration)
        DESCRIPTION: dictionary of metal and its converted ppm concentration

    """
    ppms_oxides = {}
    
    #get the concentration in ppm for each metal
    for n, oxide in enumerate(oxides):
        metal, ppm = oxide_ppm_conv(masspercs[n], oxide)
        ppms_oxides[metal] = ppm
    
    return ppms_oxides


def oxides_ppm_convfile(masspercs, oxides, filename="Oxide_ppm_concentrations.txt"):
    """
    Convert mass percentages of oxides to ppm concentrations in txt/csv file

    Parameters
    ----------
    masspercs : TYPE, list(floats)
        DESCRIPTION: list of oxide masspercentages
    oxides : TYPE, list(str)
        DESCRIPTION: list of oxides
    filename : TYPE,str
        DESCRIPTION: filename for created file, DEFAULT is "Oxide_ppm_concentrations.txt".

    Returns
    -------
    None.

    """
    filetype = filename.split(".")[-1]
    separator = "\t" if filetype == "txt" else ","
    
    #get ppm and metal for each oxide + correct format to write
    ppm_oxides = oxides_ppm_convs(masspercs, oxides)
    lines = [metal + separator + str(ppm_oxides[metal]) + "\n" for metal in ppm_oxides.keys()]
    
    #create file
    f = open(filename, "w")
    
    #add heading + write to file
    f.write("Metal" + separator + "ppm \n")
    f.writelines(lines)
    
    f.close()
    
    return None


def oxides_ppm_filetofile(oxidefile, ppmfile="ppm_concentrations.txt"):
    """
    Convert oxide masspercentages to ppm concentrations of metal from txt/csv file and write to new txt/csv file

    Parameters
    ----------
    oxidefile : TYPE, str
        DESCRIPTION: path + file name wherein oxide name and mass percentages are given
    ppmfile : TYPE, str
        DESCRIPTION: file name of newly created file with ppm concentrations, 
        DEFAULT is "ppm_concentrations.txt".

    Returns
    -------
    None.

    """
    #open file with oxide mass fractions
    print("Reading {oxidefile}...", end='')
    filetypes = [oxidefile.split(".")[-1], ppmfile.split(".")[-1]]
    separator_oxide = "\t" if filetypes[0] == "txt" else ","
    separator_ppm = "\t" if filetypes[-1] == "txt" else ","
    oxides = open(oxidefile, 'r')
    
    print("Read")
    print("Writing {ppmfile}...", end='')
    
    #write txt file for ppm concs
    with open(ppmfile, 'w') as f:
        f.write("Metal" + separator_ppm + "ppm \n")
        for line in oxides.readlines()[1:]:
            oxide, massperc = line.rstrip().split(separator_oxide)
            metal, ppm = oxide_ppm_conv(float(massperc), oxide)
            f.write(metal + separator_ppm + str(ppm) + "\n")
    oxides.close()
    print("Done")
    
    return None


"""
Rotation setup
"""



def pad_img(img, RGB=False, cte=None):
    """
    pad image with zeros to ensure rotation without loss of original data

    Parameters
    ----------
    img : TYPE, 2d list
        DESCRIPTION: 2d list of image
    RGB : TYPE, boolean
        DESCRIPTION: boolean of image intensities, False for grayscale, True for RGB
        DEFAULT is False

    Returns
    -------
    enlarged_img : TYPE, 2d list
        DESCRIPTION: padded image array

    """
    if cte is None:
        cte = -1
    #img = 2d array in grayscale or rgb values
    #check if image is grayscale or RGB 
    if len(img.shape) == 3:
        RGB = True
    #look at how big image needs to be to properly be rotated without loss of info
    H, L = img.shape[:2]
    diag = int(np.sqrt(H**2+L**2))
    
    pad_y = diag - H
    pad_top = (pad_y  // 2) 
    pad_bottom = pad_y - pad_top
    
    pad_x = diag - L
    pad_right = pad_x // 2
    pad_left = pad_x - pad_right
    
    
    
    if RGB:
        enlarged_img = np.pad(img, pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='constant', constant_values=cte)
    
    else:
        enlarged_img = np.pad(img, pad_width = ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=cte)
        
    return enlarged_img





def rotationtry(img):
    """
    try for rotation, check for loss of original data

    Parameters
    ----------
    img : TYPE, 2d list
        DESCRIPTION: 2d array from image

    Returns
    -------
    None.

    """
    #limg = pad_img(img)
    for i in np.arange(0,360,5):
        sample = skt.rotate(img, i, order=0, resize=True, preserve_range=True)
        plt.title(f'sample rotated over {i} degrees')
        plt.imshow(sample)
        plt.show()
        
    return None






"""
trial data
"""

path = "/Users/" + pc + "/OneDrive - UGent/master 2/thesis/ct data/trial images/spherile/" #changes depending on where the data is stored
path2 = "/Users/" + pc + "/OneDrive - UGent/master 2/thesis/ct data/trial images/apatite/"
path3 = "/Users/"+ pc + "/OneDrive - UGent/master 2/thesis/ct data/trial images/random/"



path_trialfiles = "/Users/" + pc + "/OneDrive - UGent/master 2/thesis/code/trials/"

path_csv = "C:/Users/" + pc + "/OneDrive - UGent/master 2/thesis/code/"

ct_data = ["Spherile_2_000250.tif", "Spherile_2_000251.tif", "Spherile_2_000252.tif"]
apatite_data = "BS95-AB0033.tif"

trial_data = np.arange(0,40, 1).reshape(4,10)
trial_dataset = [trial_data + 5*x for x in np.arange(0,10,1)]

minerals_file = "minerals.csv"
mineralphase = create_minerals(path_csv, minerals_file, 21)

trial_pixels = pixelisation(array=trial_data)


