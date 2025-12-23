# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 11:06:17 2025

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


import tifffile as tf
import pandas as pd
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import tomopy
from PIL import Image
import xraylib as xrl
import h5py
import pyvista as pv
pv.set_jupyter_backend(None)




"""
Classes
"""

class mineral_phase(object):
    def __init__(self, name, density, major_composition):
        """
        Initialise mineral_phase class

        Parameters
        ----------
        name : str
            Name of mineral phase
        density : float
            Density of mineral phase in g/cm3
        major_composition : list(str)
            List of bruto formula for mineral phase

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
    
    def add_major_composition(self, major_composition):
        self.major_composition = major_composition
        return self.major_composition
        
    

class pixels(object):
    
    def __init__(self, index, values, mineral_phase, shape, Ps_cor):
        """
        Initialise pixels class

        Parameters
        ----------
        index : int
             Index of pixel
        values : 2d array
             2D array of original image values
        mineral_phase : mineral_phase class
             Mineral phase of pixel, based on linear attenuation coefficient
        shape : tuple(int)
            Tuple of values shape
        Ps_cor : float
            Gives the correlation between a pixel and its size in real life (1 value = ps_cor µm)

        Returns
        -------
        None.

        """
        
        self.index = index # index of the pixel, row by row starting from 0
        self.values = values # 2d array of values
        self.phase = mineral_phase #mineral phase of the pixel
        self.shape = shape #shape of the pixel
        self.conc = mineral_phase.composition #concentration of each element, used for iterative quantification
        self.cor = Ps_cor # value to real size correlation


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
        

        Parameters
        ----------
        index : int
            Index of voxel
        values : 3d array
            3D array of original 3D image values
        mineral_phase : mineral_phase class
            Mineral phase of voxel, based on linear attenuation coefficient
        shape : tuple(int)
            Tuple of values shape
        Ps_cor : float
            Gives the correlation between a pixel and its size in real life (1 value = ps_cor µm)

        Returns
        -------
        None.

        """
        
        self.index = index # index of the voxel, row by row starting from 0
        self.values = values # 3d array of values
        self.phase = mineral_phase #mineral phase of the voxel
        self.shape = shape #shape of the voxel
        self.conc = mineral_phase.composition #concentration of each element, used for iterative quantification
        self.cor = Ps_cor # value to real size correlation


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
    
    def conc(self):
        return self.conc



"""
imaging
"""

def get_metadata(image):
    """
    Gets the metadata from a tiff image file

    Parameters
    ----------
    image : tiff image file
        Print the metadata from the tiff file

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
    path_ct : str
        Path to where image file is stored
    image : str
        Image file name
    array : 2d list
        Optional parameter for code tests, DEFAULT is None

    Returns
    -------
    img_array : 2d np array
        2D np array of image values

    """
    #open image and convert to 2d numpy array
    img = Image.open(path_ct+image) if path_ct and image != None else None
    img_array = np.array(img) if img != None else array
    
    return img_array


def ct_scans(path_ct=None, images=None,arrays=None):
    """
    Convert multiple ct scans (tif/tiff images) to 2d np arrays

    Parameters
    ----------
    path_ct : str
        Path to where image files are stored
    images : list(str)
        List of image file names
    arrays : 3d array
         Optional 3d array for code testing, DEFAULT IS None

    Returns
    -------
    ims : 3d np array 
        3D np array of ct scans 

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


def display_ct_scan(path_ct=None, image=None, array=None):
    """
    Displays a tif/tiff image in grayscale

    Parameters
    ----------
    path_ct : str
        Path to where image file is stored
    image : str
        Image file name
    array : 2d array
        Optional parameter for code tests, DEFAULT is None

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


def display_ct_scans(path_ct=None, images=None, arrays=None):
    """
    Displays multiple ct scans in grayscale 

    Parameters
    ----------
    path_ct : str
        Path to where the image files are stored
    images : list(str)
        List of names of image files (tif/tiff)
    arrays : 3d array
        Optional parameter for code tests, DEFAULT is None

    Returns
    -------
    None.

    """
    
    
    #displays multiple ct scans akin to display_ct_scan
    ims = ct_scans(path_ct, images) if path_ct and images != None else arrays
    
    for im in ims:
        
        #plot 2d array grayscaled
        fig = plt.figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        plt.imshow(im)
        ax.set_aspect('equal')
        
        plt.axis("off")
        plt.style.use('grayscale')
        plt.show()
        
    return None


def display_ct_scans_3d(path_ct=None, images=None, arrays=None,opacity=0.1, screenshot=False, name=None):
    """
    Displays multiple ct scans within a 3D image

    Parameters
    ----------
    path_ct : str
        Path to where the image files are stored
    images : list(str)
        List of names of image files (tif/tiff)
    arrays : 3d array
        Optional parameter for code tests, DEFAULT is None

    Returns
    -------
    None.

    """
    #displays multiple ct scans akin to display_ct_scan
    ims = ct_scans(path_ct, images) if path_ct and images != None else arrays
    
    #create 3d grid with points
    grid = pv.ImageData()
    grid.dimensions = ims.shape
    grid.point_data['values'] = ims.flatten(order="F") #vtk uses fortran order 
    
    #plot gridpoints
    p = pv.Plotter()
    p.add_volume(grid, cmap='viridis', opacity=opacity)
    p.show()
    if screenshot is True:
        p.screenshot(name)
        
    return None


"""
Pixelisation/Voxelisation
"""

def pixelisation(path='', image='', array=[], cols=2, rows=2):
    """
    Divide image or 2d array into pixels of chosen size

    Parameters
    ----------
    path : str
        Path to where image file is stored
    image : str
        File name of to be opened image
    array : 2d array
        Optional parameter to test code, DEFAULT is []
    cols : int
        Amount of columns the pixels should be in size, DEFAULT is 2
    rows : int
        Amount of rows the pixels should be in size, DEFAULT is 2

    Returns
    -------
    Pixels :  3d array
        3D array of pixels, which are each 2d arrays

    """
    #open image and convert to 2d array
    im = Image.open(path+image) if path and image != '' else None
    img_array = np.array(im) if im != None else array
    
    #check if possible to divide image into chosen pixel size
    assert len(img_array) % rows == 0 and len(img_array) // rows != 0, "Error: image size cannot be properly divided into chosen row size"
    assert len(img_array[0]) % cols == 0 and len(img_array[0]) // cols != 0, "Error: image size cannot be properly divided into chosen column size"
    
    
    #calculate amount of sections for chosen pixel size
    cols_ind, rows_ind = len(img_array[0]) // cols, len(img_array) // rows
    
    #divide original 2d array into 4d array
    columns =  np.array(np.hsplit(img_array, cols_ind))
    pix = np.array(np.hsplit(columns, rows_ind))
    
    #rearrange created 4d array into appropriate 3d array
    pixels = [pix[i][j] for i in range(rows_ind) for j in range(cols_ind)]
    return np.array(pixels)


def rejoin(pixels, size_cols):
    """
    Rejoins 3d array of pixels (2d array) into original 2d array 

    Parameters
    ----------
    pixels : 3d array of 2d arrays
        3D array of pixels given from pixelisation function
    size_cols : int
        Amount of columns of original image

    Returns
    -------
    full : 2d array
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
    Rejoins dictionary(pixel class) pixels (2d array) into original 2d array with mineral phases as intensities 

    Parameters
    ----------
    pixels : dict(index:pixels class)
        Dictionary of pixels class akin to those create by create_pixels
    size_cols : int
        Amount of columns of original image
    viridis  :  boolean
        If colormap should be in viridis scale or not, DEFAULT is FALSE

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
    
    print("Beginning reconstruction...", end='')
    
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
    print("Done")
    return None


def voxelisation(path_ct=None, images=None, arrays=None, rows=2, cols=2, depth=2): 
    """
    Divide images or 3d array into voxels of chosen size

    Parameters
    ----------
    path_ct : str
        Path to where image files are stored
    images : str
        File names of to be opened images
    arrays : 3D arrays
        Optional 3d array to test code, DEFAULT is None
    rows : int
        size of voxels in x direction, DEFAULT is 2
    cols : int
        size of voxels in y direction, DEFAULT is 2
    depth : int
        size of voxels in z direction, DEFAULT is 2

    Returns
    -------
    Voxels : 4d array (3d array)
        4D array of 3d arrays (voxels)

    """

    #rows=x, columns = y, depth = z
    #open images and convert to 3d array
    ims = ct_scans(path_ct=path_ct, images=images, arrays=arrays)
    
    #check if possible to divide image into chosen pixel size
    try:
        assert len(ims) % depth == 0 and len(ims) // depth != 0, "Error: image size cannot be properly divided into chosen depth size"
        assert len(ims[0]) % rows == 0 and len(ims[0]) // rows != 0, "Error: image size cannot be properly divided into chosen row size"
        assert len(ims[0,0]) % cols == 0 and len(ims[0,0]) // cols != 0, "Error: image size cannot be propoerly divided into chosen column size"
    
    except AssertionError as error:
        print(error)
    
    
    #calculate amount of sections for chosen pixel size
    cols_ind, rows_ind, depth_ind = len(ims[0,0]) // cols, len(ims[0]) // rows, len(ims) // depth
    
    #divide original 3d array into 5d array
    columns =  np.array(np.split(ims, cols_ind, axis=2))
    pix = np.array(np.split(columns, rows_ind, axis=2))
    vox = np.array(np.split(pix, depth_ind, axis=2))
    
    #rearrange 5d array into correct 4d array voxels
    Voxels = np.array([vox[i,j,k] for i in range(depth_ind) for j in range(rows_ind) for k in range(cols_ind)])
    
    
    return Voxels
    

def rejoin_vox(V, image_shape):
    """
    Rejoins 4d array of voxels into original 3d image

    Parameters
    ----------
    V : 4d array
        4D array of voxels, generated through voxelisation function
    image_shape : tuple(int)
        Tuple of original 3d image shape

    Returns
    -------
    full : 3d array
        3D array of original image

    """
    
    #generate 3d array of original image size
    datapoints = np.prod(image_shape) #total amount of data points
    
    grid_size = image_shape[1]*image_shape[2] # rows x column size
    height = datapoints // grid_size # returns the amount of rows given a rectangular image given
    rows = image_shape[1]
    
    full = np.array([[np.zeros(image_shape[2]) for _ in range(rows)] for _ in range(height)]) #create 3d array with correct dimensions

    #rearrange voxels into correct positions
    depth, rows, cols = 0, 0, 0
    
    for vox in V:
        vox_shape = vox.shape # 1st is depth, 2nd is rows, 3rd is cols
        
        cols += vox_shape[2]
        if cols > image_shape[2]:
            cols = vox_shape[2]
            rows += vox_shape[1]
        
        if rows >= image_shape[1]:
            rows = 0
            depth += vox_shape[0]
        
        for j, row in enumerate(vox):
            for n, val in enumerate(row):
                full[depth + j, rows + n, (cols - vox_shape[2]): cols] = val
    
    
    return np.array(full)
    
    # !!! Doesnt work for large imagesets due to cube creation for each voxel -> n^3 time
def rejoin_vox_grid(Voxels, grid_shape):
    """
    Rejoins dictionary(voxel class) of voxels (3d array) into original 3d array with mineral phases as intensities

    Parameters
    ----------
    Voxels : dict(index: voxels class)
        Dictionary of voxels, created through create_voxels function
    grid_shape : tuple(int)
        Tuple with shape of grid slice from the 3d image

    Returns
    -------
    None.

    """
    print("----------\nBeginning reconstruction\n----------")
    #reconstruct and crop voxel grid
    full = rows_voxels(Voxels, grid_shape)
    mask = full != Voxels[-1]
    coords = np.argwhere(mask)
    (z0, x0, y0), (z1, x1, y1) = coords.min(0), coords.max(0)
    
    cropped = full[z0:z1+1, x0:x1+1, y0:y1+1]
    
    
    
    #get categorical data to plot from cropped voxels
    phase_names = np.unique([v.phase.name for v in np.ravel(cropped)])
    phase_to_label = {phase : i for i, phase in enumerate(phase_names)}
    label_to_phase = {i: phase for i, phase in enumerate(phase_names)}
    
    
    phasegrid = np.vectorize(lambda v: phase_to_label[v.phase.name])(cropped)
    cmap = plt.get_cmap("tab20")

    phase_nums = np.unique(phasegrid)
    phase_colors = [cmap(v / 10)[:3] for v in phase_nums]

    #plot the voxels
    grid = pv.ImageData()
    grid.dimensions = np.array(phasegrid.shape) + 1
    
    grid.spacing = (1,1,1)
    
    grid.cell_data["phase"] = phasegrid.flatten(order="F")
    
    p = pv.Plotter()
    spacing = 1
    for idx, val in np.ndenumerate(phasegrid):
        x, y, z = idx
        cube = pv.Cube(center=(x+0.5, y+0.5, z+0.5), x_length=spacing, y_length=spacing, z_length=spacing)
        p.add_mesh(cube, color=phase_colors[val], opacity=0.7)
        
        
    legend_entries = [(label_to_phase[index], color) for index, color in enumerate(phase_colors)]
    p.add_legend(legend_entries)

    
    p.show()
    
    print("----------\nFinished reconstruction\n----------")
    
    return None


"""
mineral group functions
"""

def def_minerals(path, csv):
    """
    Create dictionary of inserted mineral phases 

    Parameters
    ----------
    path : str
        Path to csv file 
    csv : str
        File name of csv file

    Returns
    -------
    minerals : dict(mineralphase name: mineral_phase class)
        Dictionary of mineralphase classes

    """
    #open csv with mineral phase data
    minerals = {}
    data = pd.read_csv(path+csv)
    
    #create mineralphase class for each mineral phase in csv doc
    for i, phase in enumerate(data['Name']):
        minerals[phase] = mineral_phase(phase,float(data['Density'].iloc[i]), data['Major Compositions'].iloc[i])
    
    return minerals


def composition(path, csv, mineral_phase): #could be added to original def_minerals function
    """
    Add composition of each element to each inserted mineral phase

    Parameters
    ----------
    path : str
        Path to csv file
    csv : str
        Filename of csv file
    mineral_phase : dict(mineralphase name: mineralphase class)
        Dictionary of mineralphases inserted

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
    mineral : dict(mineralphase name: mineralphase class)
        Dictionary of inserted mineral phases
    energy : float
        Energy of x-ray beam used for xrf/ct measurement
    err : float
        Error to be taken on linear attenuation coeff, DEFAULT IS 5%

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


def calculate_major_composition(mineral):
    """
    Calculate a more precise major composition, using the given full composition in ppm

    Parameters
    ----------
    minerals : dict(mineralphase name: mineralphase class)
        Dictionary of inserted mineral phases

    Returns
    -------
    None.

    """
    for phase in mineral.values():
        major_comp = []
        for element in phase.composition.keys():
            conc = ppm_to_weightfrac(phase.composition[element]) / 100 #go from % to float
            if conc >= 1e-4:
                major_comp.append(element)
                major_comp.append(str(conc))
        if phase.name == "Dummy":
            major_comp = ["H", "1.0"]
            
        str_repr = "".join(major_comp)
        phase.add_major_composition(str_repr)
        
    return None
        

def create_minerals(path, csv, energy, err=5, calculate_major=False):
    """
    Create dictionary of mineralphases with attenuation coeff ranges and compositions

    Parameters
    ----------
    path : str
        Path to where csv file is located
    csv : str
        Csv file name with mineralphase data
    energy : float
        Energy of x-ray beam used in xrf/ct measurement
    err : float
        Percentual error for calculation of linear attenuation coefficient range, DEFAULT is 5%
    calculate_major : boolean
        True if necessary to calculate precise major composition using given compositions, DEFAULT is False

    Returns
    -------
    minphases : dict(mineralphase name: mineralphase class)
        Dictionary of mineralphases from csv

    """
    #create mineral class objects and add attenuation range + composition
    minphases = def_minerals(path, csv)
    composition(path, csv, minphases)
    
    if calculate_major is True:
        calculate_major_composition(minphases)
        
    minphase_cs_width(minphases, energy, err=err)
    
    return minphases


"""
pixels/voxels functions
"""

def def_pixels(pix, seen_min, minerals,  ps_cor): 
    """
    Create dictionary of pixels class

    Parameters
    ----------
    pix : 3d array
        3D np array of pixel values
    seen_min : 1d array
        Array of seen mineralphases ordered per pixel index
    ps_cor : float
        Pixel to real size correlation, in value to µm

    Returns
    -------
    Pixs : Dict(index, pixels class)
        Dictionary of pixels class with pixel values, mineralphases and correlation to real sizes

    """
    
    #starting index from 0; pix = 2d np array
    Pixs = {}
    thickness = thickness_conversionfactor(ps_cor)
    for i, val in enumerate(pix):
        Pixs[i] = pixels(i, val, seen_min[i], val.shape, thickness)
    
    #add dummy pixel -> necessary for rotation
    dummy_value = -1
    Pixs[dummy_value] = pixels(dummy_value, dummy_value, minerals["Dummy"] , (1,1), thickness )
    
    
    return Pixs


def create_pixels(data, mineralphases, path_img=None, image=None, array=[], pixel_size_y=2, pixel_size_x=2,path_min='', csv='', energy=21, ps_cor=1, err=5, calculate_major=False):
    """
    Create dict(index: pixels class) with adjusted ct scans 

    Parameters
    ----------
    data : 1d array
        1D array of grayscaled datapoints for each phase
    mineralphases : 1d array
        Names of mineralphases, corresponding to data
    path_img : str
        Path to image file location
    image : str
        Image file name
    array : 3d array
        Optional parameter for code tests, DEFAULT IS []
    pixel_size_y : int
        Amount of rows the pixels should be in size, DEFAULT is 2
    pixel_size_x : int
        Amount of columns the pixels should be in size, DEFAULT is 2
    path_min : str
        Path to csv file with mineralphase data
    csv : str
        File name of csv with mineralphase data
    energy : float
        Energy of used x-ray beam in xrf/ct measurement, DEFAULT is 21 keV
    ps_cor : float
        Pixel to real size correlation coefficient, DEFAULT is 1 pixel: 1 µm
    err: float
        Percentual error to calculate linear attenuation coefficient ranges, DEFAULT is 5%
    calculate_major : boolean
        True if necessary to calculate precise major composition using given compositions, DEFAULT is False

    Returns
    -------
    pixs : dict(index: pixel class)
        Dictionary with indexes of pixels and corresponding pixels class

    """
    
    #energy in keV
    print("Initialising mineral phases...", end='')
    mineral_phases = create_minerals(path_min, csv, energy, err, calculate_major=calculate_major)    
    
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


def def_voxels(Voxls, seen_min, minerals,  ps_cor): 
    """
    Create dictionary of voxel class

    Parameters
    ----------
    pix : 4d array
         4D np array of pixel values
    seen_min : 1d array
        1D array of seen mineralphases ordered per voxel index
    ps_cor : float
        Voxel to real size correlation, voxel to µm

    Returns
    -------
    Pixs : dict(index, voxels class)
        Dictionary of voxel class with voxel values, mineralphases and correlation to real sizes

    """
    
    #starting index from 0; pix = 2d np array
    Vox = {}
    thickness = thickness_conversionfactor(ps_cor)
    for i, val in enumerate(Voxls):
        Vox[i] = voxels(i, val, seen_min[i], val.shape, thickness)
    
    #add dummy pixel -> necessary for rotation
    dummy_value = -1
    Vox[dummy_value] = voxels(dummy_value, dummy_value, minerals["Dummy"] , (1,1,1), thickness)
    
    
    return Vox


def create_voxels(data, mineralphases, path_img=None, image=None, array=[], rows=2, cols=2, depth=2 ,path_min='', csv='', energy=21, ps_cor=1, err=5, calculate_major=False):
    """
    Create dict(index: voxels class) with adjusted ct scans 

    Parameters
    ----------
    data : 1d array
        1D array of grayscaled datapoints for each phase
    mineralphases : 1d array
        Names of mineralphases, corresponding to data
    path_img : str
        Path to image file location
    image : str
        Image file name
    array : 4d list
        Optional parameter for code tests, DEFAULT IS []
    rows :  int
        Amount of rows the pixels should be in size, DEFAULT is 2
    cols : int
        Amount of columns the pixels should be in size, DEFAULT is 2
    path_min : str
        Path to csv file with mineral_phase data
    csv : str
        File name of csv with mineralphase data
    energy : float
        Energy of used x-ray beam in xrf/ct measurement, DEFAULT is 21 keV
    ps_cor : float
        Voxel to real size correlation coefficient, DEFAULT is 1 voxel: 1 µm
    err: float
        Percentual error to calculate linear attenuation coefficient ranges, DEFAULT is 5%
    calculate_major : boolean
        True if necessary to calculate precise major composition using given compositions, DEFAULT is False

    Returns
    -------
    V : voxels class
        Dictionary with indexes of pixels and corresponding voxels class

    """
    
    #energy in keV
    print("Initialising mineral phases...", end='')
    mineral_phases = create_minerals(path_min, csv, energy, err, calculate_major=calculate_major)    
    
    print('Done')
    print("Rescaling image values...", end='')
    
    #using lin regress turn image grayscale values into lin atten coeff range
    imgs = ct_scans(path_img, image) if path_img and image is not None else array
    slope, intercept = linregress(data, mineralphases,mineral_phases, energy=energy)[:2]
    img_cor = (imgs - intercept)/ slope
    
    print("Done")
    
    #create pixels
    print("Generating voxels...", end='')
    voxl = voxelisation(arrays=img_cor, rows=rows, cols=cols, depth=depth)
 
    #add mineral phases to pixels and create pixel class objects
    
    seen_minerals = seen_min(voxl, mineral_phases)
    V = def_voxels(voxl, seen_minerals, mineral_phases, ps_cor)
    print("Done")
    return V


"""
helper functions on pixels/voxels
"""

def thickness_conversionfactor(thickness):
    """
    Converts given thickness in µm to cm    
    
    Parameters
    ----------
    thickness : float
        thickness in µm

    Returns
    -------
    Converted_thickness : float
        Converted thickness from µm to cm

    """
    return thickness * 1e-4


def seen_min(pixels, minerals): 
    """
    Create list of mineralphases for each pixel given

    Parameters
    ----------
    pixels : 3d/4d array
        3D/4D array given by pixelisation/voxelisation function
    minerals : dict(mineralphase name: mineralphase class)
        Dictionary of mineralphases with inserted mineralphase data

    Returns
    -------
    seen_min : 1d array
        Array of seen mineralphases based on linear attenuation coeff range

    """
    
    #pixels = 2d np array and minerals = dict minerals class
    #voxels = 3d np array and minerals = dict minerals class
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

"""
image data processing functions
"""


def display_histogram(img, bar=False, scale='log'):
    """
    Plots a barplot or intensity chart of image intensity values

    Parameters
    ----------
    img : 2d/3d array
        2D/3D array of original image
    bar : boolean
        Describe if barplot or intensity plot should be given, DEFAULT is False
            barplots significantly slows down calculations
    scale: str
        Scaling of y axis, DEFAULT is logarithmic

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
        plt.ylabel('Frequency')
        plt.xlabel('Pixel value')
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
    Find the peaks in a smoothened + filtered histogram plotted via display_histogram
        Assumes Gaussian peaks

    Parameters
    ----------
    img : 2d/3d array
        Image of multiple mineral phases
    nphases : int
        Amount of phases that can be seen in image
    sigmas : int or sequence
        Estimates of std of gaussian distributions in histogram
    width : int or sequence
        Width for scipy wavelet peak finding function, DEFAULT is 1000.
    window : int (odd)
        Size of window for median filtereing, DEFAULT is 3.

    Returns
    -------
    phases_peaks : 1d array
        Peak intensity average for each mineral phase

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
    Perform linear regression, based on background and standard known phases in tif/tiff image

    Parameters
    ----------
    data : 1d array
        Array of dataopoints from grayscaled image
    mineralphases_present : 1d array
        Array of mineralphases that represent the datapoints
    mineralphases : dict(name : mineralphase class)
        Dictionary of mineralphases that have been added to csv file
    energy : float
        Energy of used x-ray beam in xrf/ct measurement, DEFAULT is 21 keV

    Returns
    -------
    slope : float
        Slope of the regression line
    intercept : float
        Intercept of the regression line
    r : float
        The Pearson correlation coefficient
    p : float
        The p-value for a hypothesis test whose null hypothesis is that the slope is zero 
        using Wald Test with t-distribution
    sterr : float
        Standard error of the estimated intercept, under assumption of residual normality

    """
    mus = [xrl.CS_Energy_CP(mineralphases[x].major_composition, energy)*mineralphases[x].density for x in mineralphases_present]
    slope, intercept, r, p, sterr = sc.stats.linregress(mus, data)
    
    return slope, intercept, r, p, sterr


"""
functions on pixels
"""

def nlayers_x(Pixels, traversed_pixelnr, image_size, pixel_size):
    """
    Calculate the pixels before the traversed pixels

    Parameters
    ----------
    Pixels : pixels class
        Pixels created through create_pixels function
    traversed_pixelnr :  int
        Index of traversed pixel
    image_size : tuple(int)
        Tuple with size of original image (rows, columns)
    pixel_size : Tuple(int)
        Tuple with size of pixels (rows, columns)

    Returns
    -------
    traversed_x : 1d array
        Array with pixels that come before inputted pixel index per row

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
    Calculate the pixels above the traversed pixel

    Parameters
    ----------
    Pixels : pixels class
        Pixels created through create_pixels function
    traversed_pixelnr : int
        Index of traversed pixel
    image_size : tuple(int)
        Tuple with size of original image (rows, columns)
    pixel_size : tuple(int)
        Tuple with size of pixels (rows, columns)

    Returns
    -------
    traversed : 1d array
        Array with pixels that are directly above the inputted pixel

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
    Generate a pixelmatrix

    Parameters
    ----------
    Pixels : dict(index : pixels class)
        Pixels created through create_pixels function
    image_size : tuple(int)
        Tuple with size of original image (rows, columns)
    pixel_size : tuple(int)
        Tuple with size of original image, DEFAULT is pixel[0].shape

    Returns
    -------
    rows : 2d array
        2D array of pixels class

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
    Generate indexmatrix

    Parameters
    ----------
    Pixels : ditc(index : pixels class)
        Pixels created through create_pixels function
    image_size : tuple(int)
        Tuple with size of original image (rows, columns)
    pixel_size : tuple(int)
        Tuple with size of original image, DEFAULT is pixel[0].shape

    Returns
    -------
    rows : 2d array
        2D array of pixel indexes

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


def pad_img(img, RGB=False, cte=None):
    """
    Pad image with specified value to ensure rotation without loss of original data

    Parameters
    ----------
    img : 2d array
        2d array of image
    RGB : boolean
        Boolean of image intensities, False for grayscale, True for RGB
            DEFAULT is False

    Returns
    -------
    enlarged_img : 2d array
        Padded image array

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

"""
functions on voxels
"""

def rows_voxels(Voxels, image_size):
    """
    Generate voxel matrix cube

    Parameters
    ----------
    Voxels : dict(index : voxels class)
        Voxels created through create_voxels function
    image_size : tuple(int)
        Tuple with size of original image (depth, rows, columns)

    Returns
    -------
    rows : 3d array
        3D array of voxels 

    """
    indexmatrix = rows_index_3d(Voxels, image_size)
    pixelmatrix = []
    for grid in indexmatrix:
        g = []
        for row in grid:
            r = [Voxels[index] for index in row]
            g.append(r)
        pixelmatrix.append(g)
    return np.array(pixelmatrix)


def rows_index_3d(Voxels, images_size):
    """
    Generate index matrix cube for voxels

    Parameters
    ----------
    Voxels : dict(index : voxels class)
        Voxels created through create_voxels function
    image_size : tuple(int)
        Tuple with size of original image (depth, rows, columns)

    Returns
    -------
    padded_rows : 3d array
        3D array of voxel indexes

    """
    #calculate length of each row
    voxel_size = Voxels[0].shape
    
    grid_length  = images_size[1] // voxel_size[1]
    row_length = images_size[2] // voxel_size[2]
    
    
    assert images_size[0] % voxel_size[0] == 0, "correct voxelisation did not occur"
    assert images_size[1] % voxel_size[1] == 0, "correct voxelisation did not occur"
    assert images_size[2] % voxel_size[2] == 0, "correct voxelisation did not occur"
    
    #append each pixel to a list before adding to rows list
    full = []
    row, ct = [], 0
    grid, index = [], 0
    for p in list(Voxels.keys())[:-1]:
        row.append(p)
        ct += 1
        
        if ct == row_length:
            grid.append(row)
            row = []
            ct = 0
            index += 1
            
            
        if index == grid_length:
            full.append(grid)
            grid = []
            index = 0
            
    
    #pad with dummy pixels to rotate later in sinogram
    padded_rows = pad_img_3D(np.array(full))
    print(f'---------- \nenlarged images shape is {padded_rows.shape}\n----------')
    
    return padded_rows


#should check for RGB values but not full nec
def pad_img_3D(img, RGB=False, cte=None):
    """
    Pad 3d image with specified value to ensure rotation without loss of original data

    Parameters
    ----------
    img : 3d array
        List of image arrays (2d)
    RGB : boolean
        boolean of image intensities, False for grayscale, True for RGB
            DEFAULT is False

    Returns
    -------
    ims : 3d array
        Padded image arrays

    """
    if cte is None:
        cte = -1
    #img = 2d array in grayscale or rgb values
    #check if image is grayscale or RGB 
    if len(img.shape) == 4:
        RGB = True
    #look at how big image needs to be to properly be rotated without loss of info
    H, L = img.shape[1:3]
    diag = int(np.sqrt(H**2+L**2))
    
    pad_y = diag - H
    pad_top = (pad_y  // 2) 
    pad_bottom = pad_y - pad_top
    
    pad_x = diag - L
    pad_right = pad_x // 2
    pad_left = pad_x - pad_right
    
    ims = []
    for im in img:
        if RGB:
            enlarged_img = np.pad(im, pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='constant', constant_values=cte)
        
        else:
            enlarged_img = np.pad(im, pad_width = ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=cte)
        
        ims.append(enlarged_img)
        
    return np.array(ims)


"""
Fundamental parameter calculation functions
"""

def production_cross_section(element, line, energy):
    """
    use xraylib to calculate the xrf production cross-section

    Parameters
    ----------
    element : str
        Symbol of element of interest
    line : int
        Xrf line macro in xraylib
    energy : float
        Energy of x-ray beam used for XRF measurement

    Returns
    -------
    prod_cs : TYPE, float
        DESCRIPTION: production cross-section of given element, line and energy

    """
    Z = xrl.SymbolToAtomicNumber(element)
    prod_cs = xrl.CS_FluorLine(Z, line, energy)
    
    return prod_cs


def Geometryfactor_2d(indexmatrix, Adet, Dsd, ps_cor, angle):
    """
    

    Parameters
    ----------
    indexmatrix : 2d array
        Matrix of pixels indices
    Adet : float
        Active area of the detector in mm^2
    Dsd : float
        Distance from sample to detector in mm
    ps_cor : float
        Size comparison between pixel thickness and real size in pix: µm
    angle : float
        Float of rotation angle in radians

    Returns
    -------
    G : 2d array
        Geometryfactor.

    """
    #get coordinate grid shape
    coords_shape = indexmatrix.shape*ps_cor
    grid_half_x = coords_shape[0] // 2
    
    
    detector_pos = np.array([0, Dsd*1000])
    detector_normal = np.array([0,-1]) #point towards sample (y decrease)
    n = detector_normal / np.linalg.norm(detector_normal)
    
    #create rotation matrix
    Rmat = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    
    
    #initialise coordinate grid
    x = np.linspace(-grid_half_x, grid_half_x , coords_shape[0])
    y = np.linspace(0, -coords_shape[1], coords_shape[1])
    
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    #rotate grid
    coords = np.stack([X,Y], axis=-1)
    rotated = coords @ Rmat
        
    Xr = rotated[..., 1]
    Yr = rotated[..., 1]
    
    #compute line of sight vectors
    vx = Xr - detector_pos[0]
    vy = Yr - detector_pos[1]

    R = np.sqrt(vx**2 + vy**2) 
    
    
    #compute cos(a)
    
    dot = n[0]*vx + n[1]*vy
    cos_psi = dot / R
    
    #clip cos_psi for safety
    cos_psi =  np.clip(cos_psi, -1.0, 1.0)
    
    #calculate geometry factor
    omega_det = (Adet*cos_psi) / (R/1000)**2
    G = omega_det / (4*np.pi)
    
    return G


def Geometryfactor_3d(indexmatrix, Adet, Dsd, ps_cor, angle, detector_side='Left'):
    """
    Calculate the geometryfactor taking into accunt the tilt angle between voxel volume and detector center,
        takes volume into account

    Parameters
    ----------
    indexmatrix : 3d array
        Matrixcube of voxels indices
    Adet : float
        Active area of the detector in mm^2
    Dsd : float
        Distance form sample to detector in mm
    ps_cor : float
        Size comparison between voxel thickness and real size in vox: µm
    angle : float
        Float of rotation angle in radians
    detector_side : str
        Side of sample volume where detector is located, "Left" and "Right" are accepted,
            DEFAULT is "Left"

    Returns
    -------
    G : 3d array
        Geometry factor

    """
    #get coordinate cube shape
    coords_shape = indexmatrix.shape*ps_cor
    
    grid_half_z = coords_shape[0] // 2
    grid_half_y = coords_shape[2] // 2
    
    if detector_side == 'Left':
        detector_pos = np.array([0, -Dsd*1000, 0])
        detector_normal = np.array([0,-1, 0]) #point towards sample (x decrease)
        n = detector_normal / np.linalg.norm(detector_normal)
    
    #calculate rotation cube
    Rz = np.array([
    [1.0,           0.0,           0.0         ],
    [0.0,  np.cos(angle), -np.sin(angle)],
    [0.0,  np.sin(angle),  np.cos(angle)]])
    
    #create sample coordinate grid/cube
    z = np.linspace(-grid_half_z, grid_half_z, coords_shape[0])
    x = np.linspace(0, coords_shape[1], coords_shape[1]) if detector_side == 'Left' else np.linspace(-coords_shape[1], 0, coords_shape[1])
    y = np.linspace(-grid_half_y, grid_half_y, coords_shape[2])
    
    X,Y,Z = np.meshgrid(x,y,z, indexing='xy')
    
    coords = np.stack([X,Y,Z], axis=-1)
    
    #rotate cube
    coords_rot = coords @ Rz.T
    
    Xr = coords_rot[...,0]
    Yr = coords_rot[...,1]
    Zr = coords_rot[...,2]
    
    #compute line of sight vectors
    vz = Zr - detector_pos[0]
    vx = Xr - detector_pos[1]
    vy = Yr - detector_pos[2]
    

    R = np.sqrt(vz**2 + vx**2 + vy**2) 
    
    
    #compute cos(a)
    dot = vz*n[0] + vx*n[1] + vy*n[2]
    cos_psi = dot / R
    cos_psi = np.clip(cos_psi, -1.0, 1.0)
    
    #calculate geometry factor
    omega_det = (Adet*cos_psi) / (R/1000)**2
    G = omega_det / (4*np.pi)
    
    return G


def Geometryfactor(Adet, Dsd):
    """
    Calculate the geometryfactor taking the sample as a point instead of matrix/cube

    Parameters
    ----------
    Adet : float
        Active detection area in mm^2
    Dsd : float
        Distance from sample to detector in mm

    Returns
    -------
    G : float
        General geometry factor

    """
    #calculate geometry factor
    omega_det = Adet / Dsd**2
    G = omega_det / (4*np.pi)
    
    return G
    


def fluorescence(Pixels, energy, image_size, pixel_size=None):
    """
    Calculate the fluorescence matrix

    Parameters
    ----------
    Pixels : dict(index : pixels class)
        Pixels created through create_pixel function
    energy : float
        Energy of x-ray beam used for xrf/ct measurement
    image_size : tuple(int)
        Tuple with image size (rows, columns)
    pixel_size : tuple(int)
        Tuple with pixel size (rows, columns)

    Returns
    -------
    absorp_total : 2d array
        2D array of calculated fluorescence


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


def w_mu(Pixels, element, energy, image_size, pixel_size=None):
    """
    calculate the weight fraction matrix for a given image

    Parameters
    ----------
    Pixels : dict(index : pixels class)
        Pixels created through create_pixel function
    element : str
        String of elemental symbol
    energy : float
        Energy of x-ray beam used for xrf/ct measurement
    image_size : tuple(int)
        Tuple with image size (rows, columns)
    pixel_size : tuple(int)
        Tuple with pixel size (rows, columns)

    Returns
    -------
    wi : 2d array
        Matrix for weight fractions over mu of pixels

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
    

def FP(density, thickness, concs, mu_o, mu_f):
    """
    Calculate and return the fundamental parameters necessary for intensity calculations via preprocessed matrix parameters

    Parameters
    ----------
    density : 2d array
        2D array of density of mineral phase per pixel
    thickness : 2d array
        2D array of thickness per pixel
    concs : 2d array
        2D array of elemental concentration per pixel
    mu_o : 2d array
        2D array of linear attenuation coeffs at original x-ray beam energy per pixel
    mu_f : 2d array
        2D array of linear attenuation coeffs at fluorescence energy per pixel
    ----------
    w_mu : 2d array
        Matrix for weight fractions over mu of pixels
    fluor_total : 2d array
        Matrix for fluorescence factor
    absorption_x_total : 2d array
        Matrix for entry absorption (x)
    absorption_y_total : 2d array
        Matrix for exit absorption (y)

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
    Preprocess necessary parameters for FPM formula

    Parameters
    ----------
    Pixels : dict(index : pixels class)
        Pixels created through create_pixel function
    indexmatrix : 2d array
        Matrix of pixels indices
    element : str
        String representation of elemental symbol
    line :  int
        Macro for xrf line in xraylib
    energy : float
        Energy of used x-ray beam for XRF/CT measurement

    Returns
    -------
    density : 2d array
        Matrix of densities per pixel
    thickness : 2d array
        Matrix of thickness per pixel (cte)
    concentration : 2d array
        Matrix of concentration of specified element per pixel
    mu_o : 2d array
        Matrix of linear attenuation coeffs at original x-ray beam energy per pixel
    mu_f : 2d array
        Matrix of linear attenuation coeffs at xrf energy per pixel

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
            conc = ppm_to_weightfrac(Pixels[index].conc[element]) / 100 #go from % to float
            mu_o_pix = xrl.CS_Total_CP(Pixels[index].phase.major_composition, energy)
            mu_f_pix = xrl.CS_Total_CP(Pixels[index].phase.major_composition, energy_xrf)
            
            
            
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
 
    
def preprocess_param_3d(Voxels, indexmatrix, element, line, energy):
    """
    Preprocess necessary parameters for FPM formula

    Parameters
    ----------
    Voxels : dict(index : voxels class)
        Pixels created through create_pixel function
    indexmatrix : 3d array
        Matrix of voxel indices
    element : str
        String representation of elemental symbol
    line :  int
        Macro for xrf line in xraylib
    energy : float
        Energy of used x-ray beam for XRF/CT measurement

    Returns
    -------
    density : 3d array
        Matrix cube of densities per pixel
    thickness : 3d array
        Matrix cube of thickness per pixel (cte)
    concentration : 3d array
        Matrix cube of concentration of specified element per pixel
    mu_o : 3d array
        Matrix cube of linear attenuation coeffs at original x-ray beam energy per pixel
    mu_f : 3d array
        Matrix cube of linear attenuation coeffs at xrf energy per pixel

    """
    density = np.zeros(indexmatrix.shape)
    thickness = np.ones(indexmatrix.shape)
    concentration = np.zeros(indexmatrix.shape)
    mu_o = np.ones(indexmatrix.shape)
    mu_f = np.ones(indexmatrix.shape)
    
    Z = xrl.SymbolToAtomicNumber(element) 
    energy_xrf = xrl.LineEnergy(Z, line) 

    for z, grid in enumerate(indexmatrix):
        for x, row in enumerate(grid):
            for y, index in enumerate(row):
                d = Voxels[index].phase.density
                t = Voxels[index].cor
                conc = ppm_to_weightfrac(Voxels[index].conc[element]) / 100 #go from % to float
                mu_o_vox = xrl.CS_Total_CP(Voxels[index].phase.major_composition, energy)
                mu_f_vox = xrl.CS_Total_CP(Voxels[index].phase.major_composition, energy_xrf)
    
    
                density[z,x,y] = d
                thickness[z,x,y] = t
                concentration[z,x,y] = conc
                mu_o[z,x,y] = mu_o_vox
                mu_f[z,x,y] = mu_f_vox
    
    return density, thickness, concentration, mu_o, mu_f
    


def FP_3d(density, thickness, concs, mu_o, mu_f, detector_side='Left'):
    """
    Calculate the fundamental parameters for a 3D matrix

    Parameters
    ----------
    density : 3d array
        3D array of densities per voxel
    thickness : 3d array
        3D array of thickness per voxel
    concs : 3d array
        3D array of elemental concentration per voxel
    mu_o : 3d array
        3D array of linear attenuation coeff at original X-ray beam energy
    mu_f : 3d array
        3D array of linear attenuation coeff at XRF energy
    detector_side : str
        Side of sample where the detector is, Left or Right accepted, DEFAULT is Left

    Returns
    -------
    w_mu : 3D array
        3D array of weightfractions over linear attenuation coeff at original X-ray beam energy
    fluor_total : 3D array
        3D array of fluorescence generated per voxel
    absorption_x_total : 3D array
        3D array of absorption for entering X-ray beam photons
    absorption_y_total : 3D array
        3D array of absorption for exiting XRF photons

    """
    
    detsides = {"Right": (slice(None, None, -1), slice(None, None, -1)), "Left": (slice(None, None, -1 ), slice(None, None, 1))}
    slicestyle = detsides[detector_side]
    
    slicing = []
    s = []
    for z in range(len(density)):
        s.append(z)
        s.append(slicestyle[0])
        s.append(slicestyle[1])
        s = tuple(s)
        
        slicing.append(s)
        s = []
    slicing  = tuple(slicing)
    
    w_mu = np.zeros(density.shape)
    fluor_total = np.zeros(density.shape)
    absorption_x_total = np.zeros(density.shape)
    absorption_y_total = np.zeros(density.shape)
    
    #calculate FP per slice -> same logic as 2d sample
    for gridindex, sliced in enumerate(slicing):
    # Calculate fundamental parameters
        w_mu[gridindex] = concs[gridindex] / mu_o[gridindex]
        
        fluor_total[gridindex] = 1 - np.exp(-mu_o[gridindex] * thickness[gridindex] * density[gridindex])
        
        # Compute cumulative absorption along X and Y
        
        absorption_x_total[gridindex] = np.cumprod(
            np.exp(-mu_o[gridindex, ::-1] * thickness[gridindex, ::-1] * density[gridindex, ::-1]), axis=0
        )[::-1]

        
        absorption_y_total[gridindex] = np.cumprod(
            np.exp(-mu_f[sliced] * thickness[sliced] * density[sliced]), axis=1
        )[slicestyle]
        
    
    return w_mu, fluor_total, absorption_x_total, absorption_y_total
    

# #do per slice -> evereything already 2d arrays
def FP_slice(density, thickness, concs, mu_o, mu_f, detector_side='Left'):
    """
    Calculate the fundamental parameters for a 3D matrix per slice

    Parameters
    ----------
    density : 2d array
        2D array of densities per voxel
    thickness : 2d array
        2D array of thickness per voxel
    concs : 2d array
        2D array of elemental concentration per voxel
    mu_o : 2d array
        2D array of linear attenuation coeff at original X-ray beam energy
    mu_f : 2d array
        2D array of linear attenuation coeff at XRF energy
    detector_side : str
        Side of sample where the detector is, Left or Right accepted, DEFAULT is Left

    Returns
    -------
    w_mu : 2D array
        2D array of weightfractions over linear attenuation coeff at original X-ray beam energy
    fluor_total : 2D array
        2D array of fluorescence generated per voxel
    absorption_x_total : 2D array
        2D array of absorption for entering X-ray beam photons
    absorption_y_total : 2D array
        2D array of absorption for exiting XRF photons

    """
    
    detsides = {"Right": (slice(None, None, -1), slice(None, None, -1)), "Left": (slice(None, None, -1 ), slice(None, None, 1))}
    slicestyle = detsides[detector_side]
    
    # Calculate fundamental parameters
    wmu_slice = concs / mu_o
    fluor_slice = 1 - np.exp(-mu_o * thickness * density)
    
    # Compute cumulative absorption along X and Y
    
    abs_x_slice = np.cumprod(
        np.exp(-mu_o[::-1] * thickness[::-1] * density[::-1]), axis=0
    )[::-1]
    
    
    abs_y_slice = np.cumprod(
        np.exp(-mu_f[slicestyle] * thickness[slicestyle] * density[slicestyle]), axis=1
    )[slicestyle]
    
    
    return wmu_slice, fluor_slice, abs_x_slice, abs_y_slice

"""
Intensity calculation functions
"""
#!!!
def I_fluor(I0, G, Qi, w_mu, fluormatrix, absorption_x, absorption_y): #for pixels
    """
    Calculate theorhetical intensity of specific xrf line for chosen element,
        uses fpm formula derived from tom schoonjans,
            with precalculated fpm parameters


    Parameters
    ----------
    I0 : float
        Original x-ray beam intensity in ph/s
    G : float
        Geometry factor calculated through geometryfactor function, 
            geometry factor for xrf setup (a=90°)
    Qi : float
        Production cross sections calculated through prod_cross_section function
    w_mu : 2d array
        Matrix for weight fractions over mu of pixels
    fluormatrix : 2d array
        Matrix for fluorescence factor
    absorption_x : 2d array
        Matrix for entry absorption (x)
    absorption_y : 2d array
        Matrix for exit absorption (y)

    Returns
    -------
    intensity : 1d array
        1d array of calculated theorhetical intensities for each row (entry point) for a conventional xrf setup


    """
    #calculate intensity
    matrix = w_mu*absorption_x*fluormatrix*absorption_y
    smatrix = np.sum(matrix, axis=1)#1d array
    intensity = I0*G*Qi*smatrix
    
    return intensity
    

def I_fluor_conv(I0, G, Qi, w_mu, fluormatrix, absorption_x, absorption_y): #for voxels
    """
    Calculate theorhetical intensity of specific xrf line for chosen element,
        uses fpm formula derived from tom schoonjans,
            from precalculated fpm parameters 


    Parameters
    ----------
    I0 : float
        Original x-ray beam intensity
    G : float
        Geometry factor calculate through geometryfactor function, 
            geometry factor for xrf setup
    Qi : float
        Production cross sections calculated through prod_cross_section function
    w_mu : 3d array
        Matrix cube for weight fractions over mu of pixels
    fluormatrix : 3d array
        Matrix cube for fluorescence factor
    absorption_x : 3d array
        Matrix cube for entry absorption (x)
    absorption_y : 3d array
        Matrix cube for exit absorption (y)

    Returns
    -------
    intensity : 2d array
        2d array of calculated theorhetical intensities for each entry point for a conventional xrf setup


    """
    #calculate intensity
    matrix = w_mu*absorption_x*fluormatrix*absorption_y
    smatrix = np.sum(matrix, axis=1)[::-1] #should be condensed matrix
    intensity = I0*G*Qi*smatrix
    
    return intensity
    

def I_fluor_confocal(I0, G, Qi, w_mu, fluormatrix, absorption_x, absorption_y): #for voxels
    """
    Calculate theorhetical intensity of specific xrf line for chosen element,
        uses fpm formula derived from tom schoonjans,
            from precalculated fpm parameters 


    Parameters
    ----------
    I0 : float
        Original x-ray beam intensity
    G : float
        Geometry factor calculate through geometryfactor function, 
            geometry factor for xrf setup (a=90°)
    Qi : float
        Production cross sections calculated through prod_cross_section function
    w_mu : 3d array
        Matrix cube for weight fractions over mu of pixels
    fluormatrix : 3d array
        Matrix cube for fluorescence factor
    absorption_x : 3d array
        Matrix cube for entry absorption (x)
    absorption_y : 3d array
        Matrix cube for exit absorption (y)

    Returns
    -------
    intensity : 3d array
        3d array of calculated theorhetical intensities for each entry point for a confocal xrf setup

    """
    intensity = I0*G*Qi*w_mu*absorption_x*fluormatrix*absorption_y
    
    return intensity

"""
sinogram functions
"""

def thetas(dtheta, dualdet=False):
    """
    Calculates theta angles in radians from degrees

    Parameters
    ----------
    dtheta : float
        Increments of angle in degrees
    dualdet : boolean
        If dual detector setup is used for ct-measurement (180° vs 360°), DEFAULT is False

    Returns
    -------
    theta : 1d array
        Array of uniformly distributed projection angles in radians

    """
    if dualdet is False:
        sections = int(np.ceil(360 / dtheta)+1)
    
    else:
        sections = int(np.ceil(180 / dtheta)+1)
    
    
    theta = tomopy.angles(sections, 0, 360)
    return theta



def sinogram_rotation_2d(I0, Adet, Dsd, element, line, energy, Pixels, image_size, theta, dualdet=False): #for pixels
    """
    Create a sinogram of XRF intensities calculated using FPM method for pixels

    Parameters
    ----------
    I0 : float
        Original intensity of x-ray beam in ph/s
    Adet : float
        Active detector area in mm^2
    Dsd : float
        Distance from sample to detector in mm
    element : str
        String of elemental symbol
    line : int
        Macro for chosen line in xrl
    energy : float
        Energy of x-ray beam used in the experiment
    Pixels : dict(index : pixels class)
        Pixels created through create_pixels function
    image_size : tuple(int)
        Tuple of integers, representing the image size
    theta : 1d array
        Array of angles in radians
    dualdet : boolean    
        True if dual detector setup is used (180°) else 360°, DEFAULT is False 
            
    Returns
    -------
    sinogram : 2d array
        2D array sinogram

    """
    bt = time.time()
    
    
    #create sinogram
    
    image_indexes = rows_index(Pixels, image_size)
    sinogram_shape = len(theta), image_indexes.shape[1]
    image_shape = image_indexes.shape
    sinogram  = np.zeros(sinogram_shape)
    mask = np.ones(image_shape)
    #calculate intensity for each angle in theta
    
    # FP parameters that dont change (much) with rotation angle of sample
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
        sinogram[n,:] = intensity
        
    
    et = time.time()
    print(f"Time taken for calculation sinogram: {et-bt}")
    return sinogram


#rework -> 2d slices
def sinogram_rotation_3d(I0, Adet, Dsd, element, line, energy, Voxels, image_size, theta, dualdet=False): #for voxels
    """
    Create a sinogram of XRF intensities calculated using FPM method for pixels

    Parameters
    ----------
    I0 : float
        Original intensity of x-ray beam in ph/s
    Adet : float
        Active detector area in mm^2
    Dsd : float
        Distance from sample to detector in mm
    element : str
        String of elemental symbol
    line : int
        Macro for chosen line in xrl
    energy : float
        Energy of x-ray beam used in the experiment
    Voxels : dict(index : voxels class)
        Pixels created through create_pixels function
    image_size : tuple(int)
        Tuple of integers, representing the image size
    theta : 1d array
        Array of angles in radians
    dualdet : boolean    
        True if dual detector setup is used (180°) else 360°, DEFAULT is False 
            
    Returns
    -------
    total_slices : 3d array
        3D array of 2d sinograms

    """
    bt = time.time()
    
    #create sinogram
    
    image_indexes = rows_index_3d(Voxels, image_size)
    sinogram_shape = len(theta), image_indexes.shape[2]
    total_slices = []
    sinogram  = np.zeros(sinogram_shape)
    #calculate intensity for each angle in theta
    
    # FP parameters that dont change (much) with rotation angle of sample
    G = Geometryfactor(Adet, Dsd)
    Qi = production_cross_section(element, line, energy)

    
    #calculate parameters necessary for attenuation/fluorescence matrices
    print("Calculating parameters...", end='')
    b = time.time()
    density, thickness, concentration, mu_o, mu_f = preprocess_param_3d(
                                                Voxels, image_indexes, element, line, energy)
    e = time.time()
    print(f"Done in {e-b} seconds")
    
    for nr, grid in enumerate(image_indexes):
        for n, angle in enumerate(theta):
            angle_degree = angle * (180 / np.pi)
            print(f'Calculating intensity for slice {nr+1} rotation angle: {angle_degree:.2f}°')
            
            dens, t, conc, muo, muf = density[nr], thickness[nr], concentration[nr], mu_o[nr], mu_f[nr]
            
            mask = np.ones(grid.shape)
    
            #rotate sample + mask to find distinguish non-dummy data from dummy data
            rotated = sc.ndimage.rotate(grid, angle_degree, reshape=False, order=0)
            rotated_mask = sc.ndimage.rotate(mask, angle_degree, reshape=False, order=0)
            
            rotated[rotated_mask != 1] = -1 #set to dummy value (standard = -1)
            
            rot_dens = sc.ndimage.rotate(dens, angle_degree, reshape=False, order=0, cval = 0)
            rot_conc = sc.ndimage.rotate(conc, angle_degree, reshape=False, order=0, cval=0)
            rot_mu_o = sc.ndimage.rotate(muo, angle_degree, reshape=False, order=0, cval=1)
            rot_mu_f = sc.ndimage.rotate(muf, angle_degree, reshape=False, order=0, cval=1)

            #calculate remaining fundamental parameters for this rotated image
            w_mu, fluor, absorption_x, absorption_y = FP_slice(rot_dens, t, rot_conc, rot_mu_o, rot_mu_f)
            
            
            #calculate intensity of fluorescence
            intensity = I_fluor_conv(I0, G, Qi, w_mu, fluor, absorption_x, absorption_y)
                
            #append to sinogram
            sinogram[n,:] = intensity
        
        total_slices.append(sinogram)
        sinogram = np.zeros(sinogram_shape) #make sure to reset -> shouldnt matter but just to be sure
        
    
    et = time.time()
    print(f"Time taken for calculation sinogram: {et-bt}")
    return np.array(total_slices)


def reconstruction_sinogram_2d(sinogram, theta, algo='fbp'):
    """
    Reconstruct simulated xrf intensities from sinogram

    Parameters
    ----------
    sinogram : 2d array
        XRF sinogram of image
    theta : 1d array
        Array of angles in radians
    algo : str
        Algorithm to be used in reconstruction, use supported tomopy algorithms
            DEFAULT is fbp
        

    Returns
    -------
    recon : 3d array
        Reconstructed XRF intensities

    """
    #need to make sinogram 3D for tomopy function
    sino3d = sinogram[np.newaxis, :, :]
    
    print(f"Starting reconstruction using {algo}...", end='')
    #let tomopy perform reconstruction using specified algorithm
    recon = tomopy.recon(sino3d, theta, algorithm=algo, sinogram_order=True)
    print("Done")
    return recon


def reconstruction_sinogram_3d(total_slices, theta, algo='fbp'):
    """
    Reconstruct simulated xrf intensities from sinogram

    Parameters
    ----------
    total_slices : 3d array
        Set of xrf sinograms
    theta : 1d array
        Array of angles in radians
    algo : str
        Algorithm to be used in reconstruction, use supported tomopy algorithms
            DEFAULT is fbp
        

    Returns
    -------
    recon : 3d array
        Reconstructed XRF intensities per slice

    """
    
    print(f"Starting reconstruction using {algo}...", end='')
    #let tomopy perform reconstruction using specified algorithm
    recon = tomopy.recon(total_slices, theta, algorithm=algo, sinogram_order=True)
    print("Done")
    return recon
    

"""
quantification functions
"""

# def quantify_2d(I0, Adet, Dsd, element, line, energy, Pixels, image_size, theta, experimental_sinogram, error= 5, dualdet=False):
    
#     #get concentration per pixel
#     image_indexes = rows_index(Pixels, image_size)
#     concentration = [[ppm_to_weightfrac(Pixels[index].conc[element]) / 100  for index in row] for row in image_indexes]
   
#     #create sinogram + reconstructed image based on simulation
#     sinogram  = sinogram_rotation_2d(I0, Adet, Dsd, element, line, energy, Pixels, image_size, theta)
#     reconstruction = reconstruction_sinogram_2d(sinogram, theta)
   
#     #make reconstructed image based on experimnental data
#     recon_real = reconstruction_sinogram_2d(experimental_sinogram, theta)
    
#     #calculate regression for concentration and reconstructed image intensity
#         #Assume concentration -> intensity -> reconstruction is equal for simulation and simulation
        
#!!! TO DO
#create confocal 2d intensity calc    
   
   

       
    
    
    


    
"""
weightfraction functions
"""

def ppm_to_weightfrac(ppm):
    """
    Convert ppm concentrations to weightfractions

    Parameters
    ----------
    ppm : float
        Concentration of element in ppm

    Returns
    -------
    weightfrac : float
        Concentration of element in weight fraction in %

    """
    
    weightfrac = ppm / 1e4
    
    return weightfrac


def weightfrac_to_ppm(weightfrac):
    """
    Convert weightfractions to ppm concentrations

    Parameters
    ----------
    weightfrac : float
        Concentration of element in weight fraction %

    Returns
    -------
    ppm : float
        Concentration of element in ppm

    """
    
    return 1e4*weightfrac


def oxide_ppm_conv(massper, oxide):
    """
    Convert Oxide mass fraction to ppm concentrations of metal

    Parameters
    ----------
    massper : float
        Oxide masspercentage 
    oxide : str 
        oxide brutoformula

    Returns
    -------
    metal : str
        metal symbol
    ppm : float
        Converted ppm concentration

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
    Convert multiple oxides to ppm concentrations of respective metals

    Parameters
    ----------
    masspercs : list(floats)
        Array of masspercentages of oxides
    oxides : list(str)
        Array of oxides brutoformulae

    Returns
    -------
    ppms_oxides : dict(metal: ppm concentration)
        Dictionary of metal and its converted ppm concentration

    """
    ppms_oxides = {}
    
    #get the concentration in ppm for each metal
    for n, oxide in enumerate(oxides):
        metal, ppm = oxide_ppm_conv(masspercs[n], oxide)
        ppms_oxides[metal] = ppm
    
    return ppms_oxides


def density(densities, weightfractions, ppm=True):
    weightfractions = weightfractions if ppm is False else weightfractions / 1e6
    
    #1 / rho_mix = sum( weightfraction i / density i)
    over_rho = []
    assert len(densities) == len(weightfractions), "Incorrect input, densities and weightfraction lengths don't match."
    
    for index, dens in enumerate(densities):
        frac = weightfractions[index] / dens
        over_rho.append(frac)
    
    rho_mix = 1  / np.sum(over_rho)
    
    return rho_mix


"""
filewriting functions
"""

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


def minerals(path_file, filename, path_csv, csv, energy):
    """
    Write mineralphase data to h5 file

    Parameters
    ----------
    path_file : str
        Path to h5 file to write in/create
    filename : str
        File name of h5 to create/ write in
    path_csv : str
        Path to csv file with mineralphase data
    csv : str
        File name of csv with mineralphase data
    energy : float
        Energy of used x-ray beam in xrf/ct measurement

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


