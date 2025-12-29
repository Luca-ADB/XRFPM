# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 21:04:46 2025

@author: lucadebruyn
"""

import os
import sys

#!!! insert XRFPM python file using sys.path.insert


import XRFPM
import xraylib as xrl
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import h5py

"""
file paths of mineral phase file
"""
# path_min="FILL IN WITH PATH TO .CSV FILE"
# minerals_file="FILL IN WITH NAME OF .CSV FILE"
# calibration_file="FILL IN WITH NAME OF .CSV FILE"

"""
create shapes
"""

def circle(x, y, xi, yi, radius):
    X, Y = np.arange(0,x), np.arange(0,y)
    xv, yv = np.meshgrid(X, Y)
    circle = (xv-xi)**2 + (yv-yi)**2 <= radius**2
    
    return circle 


def parallelogram(height, width, shift, val, bkg):
    rows = height
    cols = width + shift*(height-1)
    arr = [[bkg for _ in range(cols)] for _ in range(rows)]
    
    for r in range(rows):
        offset = shift * (height-r-1)
        for c in range(width):
            arr[r][offset+c] = val
    
    return arr


def sphere(x, y, z, xi, yi, zi, r):
    X,Y,Z = np.arange(0,x), np.arange(0,y), np.arange(0,z)
    xv, yv, zv = np.meshgrid(X,Y,Z)
    mask = (xv-xi)**2 + (yv-yi)**2 + (zv-zi)**2 <= r**2
    return mask
    

"""
trial - 1 -> apatite grains (2 phase)
"""
#dataset -> https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-216

#path="FILL IN WITH PATH TO .TIFF IMAGE"
#apatite_data="FILL IN WITH NAME OF .TIFF IMAGE"

# ps_cor = 2*4.58 #µm

# #trial_specific pixels
# img = XRFPM.ct_scan(path2, apatite_data)
# plt.imshow(img, cmap="gray")
# plt.colorbar()
# plt.show()

# imshape = img.shape

# #display histogram to find phases and grayscale values
# XRFPM.display_histogram(img)
# #bkg_data, std_data = XRFP.hist_to_phasedata(img, 2, 500)# -> not working because almsot all matrix and background same intensity -> gets removed by median filter
# data = [8192, 30000]
# #calculate mineral phases and pixels 
# phases = ["Oxygen", "Hydroxylapatite"]
# pix = XRFPM.create_pixels(data, phases, path2, apatite_data, path_min=path_csv, csv=minerals_file, energy=100, err=11, ps_cor=ps_cor, pixel_size_x=1, pixel_size_y=1)
# detected_phases = XRFPM.rejoin_map_ct(pix, imshape[1])

# i0 = 1e10
# Adet = 30 #mm2
# Dsd = 30 #mm
# element = 'La'
# line = 0 # ka1
# energy = 90 #kev
# im_size = img.shape
# dtheta = 1
# theta = XRFPM.thetas(dtheta)

# sino = XRFPM.sinogram_rotation_2d(i0, Adet, Dsd, element, line, energy, pix, im_size, theta)
# plt.imshow(sino)
# plt.ylabel(r"$\phi\,(°)$", fontsize=16)
# plt.xlabel(r"$Displacement\, (pixels)$", fontsize=16)
# plt.show()

# recon = XRFPM.reconstruction_sinogram_2d(sino, theta, 'art')
# plt.imshow(recon[0].T)
# cbar = plt.colorbar()

# plt.ylabel(r"$Y\, (pixels)$", fontsize=16)
# plt.xlabel(r"$X\, (pixels)$", fontsize=16)
# plt.show()


"""
trial - 2 -> simple 4 phase system
"""
# energy = 90 #kev
# Adet = 30 #mm2
# Dsd = 30 #mm
# i0 = 1e10
# element = 'La'
# dtheta = 1
# theta = XRFPM.thetas(dtheta)
# ps_cor = 1 #µm

# #create image
# im_shape = 200, 200
# im = np.full(im_shape, 2) #create matrix of C in 255 grayscale
# circle1 = circle(200, 200, 20, 30, 15)
# circle2 = circle(200, 200, 30, 150, 25)
# circle3 = circle(200, 200, 150, 75, 35)

# im_size = 200,200
# line = 0

# #put circles on im
# im[circle1] = 188
# im[circle2] = 81
# im[circle3] = 255

# #display image
# plt.imshow(im)
# plt.show()

# # #display histogram to find phases and grayscale values
# # XRFPM.display_histogram(im)
# # bkg_data, std_data = XRFPM.hist_to_phasedata(im, 4, 1) # doesnt work on clean data!!!
# data = [2,188,81,255]
# phases = ["Matrix", "Calcite","Forsterite","Chlorapatite"]

# #create pixels
# pix = XRFPM.create_pixels(data, phases, array=im, path_min=path_min, csv=minerals_file, energy=energy, err=10, ps_cor=ps_cor, pixel_size_y=1, pixel_size_x = 1)
# detected_phases = XRFPM.rejoin_map_ct(pix, 200)
# sino = XRFPM.sinogram_rotation_2d(i0, Adet, Dsd, element, line, energy, pix, im_size, theta)
# recon = XRFPM.reconstruction_sinogram_2d(sino, theta)

# plt.imshow(sino)
# plt.ylabel(r"$\phi (degrees)$", fontsize=16)
# plt.xlabel(r"$Displacement (pixels)$", fontsize=16)
# cbar = plt.colorbar()
# cbar.ax.set_title(r"$\frac{photons}{s}$", pad=10)

# plt.show()

# plt.imshow(recon[0].T)
# cbar = plt.colorbar()

# plt.ylabel(r"$Y (pixels)$", fontsize=16)
# plt.xlabel(r"$X (pixels)$", fontsize=16)
# plt.show()


"""
trial - 3 -> 2 phase system (rectangle)
"""
# energy = 21 #kev
# Adet = 300 #mm2
# Dsd = 30 #mm
# i0 = 1e6
# element = 'Fe'
# dtheta = 1
# line = 0
# theta = XRFPM.thetas(dtheta)
# ps_cor = 2 #µm

# #create image
# im_shape = 75, 50
# im = np.full(im_shape, 100) #create matrix of Fe in 255 grayscale
# pad_h = 200 - im_shape[0]
# pad_w = 200 - im_shape[1]
# im  = np.pad(im, pad_width=((pad_h//2 + 1, pad_h//2), (pad_w//2, pad_w//2)), constant_values=3)
# im_size  = im.shape


# #display image
# XRFPM.display_ct_scan(array=im)

# # #display histogram to find phases and grayscale values
# # XRFPM.display_histogram(im)
# # bkg_data, std_data = XRFPM.hist_to_phasedata(im, 4, 1) # doesnt work on clean data!!!
# data = [3, 100]
# phases = ["Oxygen", "Fayalite"]

# #create pixels
# pix = XRFPM.create_pixels(data, phases, array=im, path_min=path_min, csv=minerals_file, energy=energy, err=5, ps_cor=ps_cor, pixel_size_y = 1, pixel_size_x = 1, calculate_major=True)
# detected_phases = XRFPM.rejoin_map_ct(pix, 200)
# sino = XRFPM.sinogram_rotation_2d(i0, Adet, Dsd, element, line, energy, pix, im_size, theta)
# recon = XRFPM.reconstruction_sinogram_2d(sino, theta)

# plt.imshow(sino)
# plt.show()

# plt.imshow(recon[0].T)
# plt.show()


"""
trial - 4 -> 2 phase system (angled rectangle)
"""
# energy = 21 #kev
# Adet = 300 #mm2
# Dsd = 30 #mm
# i0 = 1e6
# element = 'Fe'
# dtheta = 1
# line = 0
# theta = XRFPM.thetas(dtheta)
# ps_cor = 2 #µm

# #create image
# im_shape = 75, 50
# im = np.full(im_shape, 100) #create matrix of Fe in 255 grayscale
# pad_h = 200 - im_shape[0]
# pad_w = 200 - im_shape[1]
# im  = np.pad(im, pad_width=((pad_h//2 + 1, pad_h//2), (pad_w//2, pad_w//2)), constant_values=3)
# im = sc.ndimage.rotate(im, 45, order=0, reshape=False, cval=3)
# im_size  = im.shape


# #display image
# XRFPM.display_ct_scan(array=im)

# # #display histogram to find phases and grayscale values
# # XRFPM.display_histogram(im)
# # bkg_data, std_data = XRFPM.hist_to_phasedata(im, 4, 1) # doesnt work on clean data!!!
# data = [3, 100]
# phases = ["Oxygen", "Fayalite"]

# #create pixels
# pix = XRFPM.create_pixels(data, phases, array=im, path_min=path_min, csv=minerals_file, energy=energy, err=5, ps_cor=ps_cor, pixel_size_y = 1, pixel_size_x = 1)
# detected_phases = XRFPM.rejoin_map_ct(pix, 200)
# sino = XRFPM.sinogram_rotation_2d(i0, Adet, Dsd, element, line, energy, pix, im_size, theta)
# recon = XRFPM.reconstruction_sinogram_2d(sino, theta)

# plt.imshow(sino)
# plt.show()

# plt.imshow(recon[0], cmap='viridis')
# plt.show()


"""
trial 5 -> 2 phase system (parallelogram)
"""
# energy = 21 #kev
# Adet = 30 #mm2
# Dsd = 30 #mm
# i0 = 1e6
# element = 'Fe'
# dtheta = 1
# line = 0
# theta = XRFPM.thetas(dtheta)
# ps_cor = 1 #µm

# #create image
# im_shape = 75, 50
# im = parallelogram(im_shape[0], im_shape[1], 1, 100, 3) #create matrix of Fe in 255 grayscale
# im = np.flip(im, axis=1)

# pad_h = 200 - im_shape[0]
# pad_w = 200 - im_shape[1]
# im  = np.pad(im, pad_width=((pad_h//2 + 1, pad_h//2), (pad_w//2, pad_w//2)), constant_values=3)

# im_size  = im.shape
# print(im_size)

# #display image
# plt.imshow(im)
# plt.show()

# # #display histogram to find phases and grayscale values
# # XRFPM.display_histogram(im)
# # bkg_data, std_data = XRFPM.hist_to_phasedata(im, 4, 1) # doesnt work on clean data!!!
# data = [3, 100]
# phases = ["Oxygen", "Fayalite"]

# #create pixels
# pix = XRFPM.create_pixels(data, phases, array=im, path_min=path_min, csv=minerals_file, energy=energy, err=10, ps_cor=ps_cor, pixel_size_y = 1, pixel_size_x = 1)
# detected_phases = XRFPM.rejoin_map_ct(pix, im_size[1])
# sino = XRFPM.sinogram_rotation_2d(i0, Adet, Dsd, element, line, energy, pix, im_size, theta)
# recon = XRFPM.reconstruction_sinogram_2d(sino, theta)

# plt.imshow(sino)
# plt.ylabel(r"$\phi (degrees)$", fontsize=16)
# plt.xlabel(r"$Displacement (pixels)$", fontsize=16)
# cbar = plt.colorbar()
# cbar.ax.set_title(r"$\frac{photons}{s}$", pad=10)
# plt.show()

# plt.imshow(recon[0].T)
# plt.ylabel(r"$Y (pixels)$", fontsize=16)
# plt.xlabel(r"$X (pixels)$", fontsize=16)
# cbar = plt.colorbar()
# plt.show()


"""
trial 6 -> 3d sphere
"""
# energy = 90 #kev
# Adet = 30 #mm2
# Dsd = 30 #mm
# i0 = 1e10
# element = 'La'
# line = 0
# dtheta = 1
# theta = XRFPM.thetas(dtheta)
# ps_cor = 1 #µm

# #create image
# im_shape = 100,100,100
# im = np.full(im_shape, 2) #create matrix of C in 255 grayscale
# sphere1 = sphere(100, 100,100, 50,20, 20, 15)
# sphere2 = sphere(100, 100,100, 50,50, 50, 25)
# sphere3 = sphere(100, 100, 100, 20,80,80, 20)

# im_size = 100,100,100
# line = 0

# #put circles on im
# im[sphere1] = 81
# im[sphere2] = 188
# im[sphere3] = 255

# #display image
# # XRFPM.display_ct_scans_3d(arrays=im, opacity=0.05)

# # #display histogram to find phases and grayscale values
# # XRFPM.display_histogram(im)
# # bkg_data, std_data = XRFPM.hist_to_phasedata(im, 4, 1) # doesnt work on clean data!!!
# data = [2,191,81,255]
# phases = ["Matrix", "Calcite","Forsterite","Hydroxylapatite"]

# #create voxels
# vox = XRFPM.create_voxels(data, phases, array=im, path_min=path_min, csv=minerals_file, energy=energy, err=20, ps_cor=ps_cor, rows=1, cols=1, depth=1)
# # detected_phases = XRFPM.rejoin_vox_grid(vox, im_size) #doesnt work because image is too large
# sino = XRFPM.sinogram_rotation_3d(i0, Adet, Dsd, element, line, energy, vox, im_size, theta)
# XRFPM.display_ct_scans_3d(arrays=sino, opacity=0.1)

# recon = XRFPM.reconstruction_sinogram_3d(sino, theta)
# XRFPM.display_ct_scans_3d(arrays=recon, opacity=0.1)
# for i in range(len(recon)):
#     plt.imshow(recon[i].T)
#     plt.title(f"reconstruction for slice {i+1}")
#     plt.show()
    

"""
trial - 7 -> 1 phase system (rectangle) -> check intensity vs theory
"""
# energy = 8 #kev
# Adet = 30 #mm2
# Dsd = 30 #mm
# i0 = 1e10
# element = 'Co'
# dtheta = 1
# line = 0
# theta = XRFPM.thetas(dtheta)
# ps_cor = 1 #µm

# #create image
# im_shape = 10, 10
# im = np.full(im_shape, 100) #create matrix of Co in 255 grayscale
# im_size = im.shape



# #display image
# plt.imshow(im)

# # #display histogram to find phases and grayscale values
# # XRFPM.display_histogram(im)
# # bkg_data, std_data = XRFPM.hist_to_phasedata(im, 4, 1) # doesnt work on clean data!!!
# data = [0, 100]
# phases = ["Matrix", "Cobalt"]

# #create pixels
# pix = XRFPM.create_pixels(data, phases, array=im, path_min=path_min, csv=minerals_file, energy=energy, err=5, ps_cor=ps_cor, pixel_size_y = 1, pixel_size_x = 1, calculate_major=True)
# detected_phases = XRFPM.rejoin_map_ct(pix, im_shape[1])
# sino = XRFPM.sinogram_rotation_2d(i0, Adet, Dsd, element, line, energy, pix, im_size, theta)
# recon = XRFPM.reconstruction_sinogram_2d(sino, theta)


# indexmatrix = XRFPM.rows_index(pix, im_size)
# pixelmatrix = XRFPM.rows_pixels(pix, im_size)
# density, thickness, conc, mu_o, mu_f = XRFPM.preprocess_param(pix, indexmatrix, element, line, energy)
# w_mu, fluor, abs_x, abs_y = XRFPM.FP(density, thickness, conc, mu_o, mu_f)
# G = XRFPM.Geometryfactor(Adet, Dsd)
# Qi = XRFPM.production_cross_section(element, line, energy)
# ik = XRFPM.I_fluor(i0, G, Qi, w_mu, fluor, abs_x, abs_y)

# print(f'the intensity for a cobalt foil with thickness of {im.shape[1]} µm is {ik.max()}')


"""
trial - 8 -> 1 phase system (rectangular foil) -> check intensity vs theory
"""
# energy = 8 #kev
# Adet = 30 #mm2
# Dsd = 30 #mm
# i0 = 1e10
# element = 'Co'
# dtheta = 1
# line = 0
# theta = XRFPM.thetas(dtheta)
# ps_cor = 1 #µm

# #create image
# im_shape = 100,10,100
# im = np.full(im_shape, 100) #create matrix of C in 255 grayscale
# im_size = im.shape


# #display image
# XRFPM.display_ct_scans_3d(arrays=im, opacity=0.05)

# # #display histogram to find phases and grayscale values
# # XRFPM.display_histogram(im)
# # bkg_data, std_data = XRFPM.hist_to_phasedata(im, 4, 1) # doesnt work on clean data!!!
# data = [0,100]
# phases = ["Matrix", "Cobalt"]

# #create voxels
# vox = XRFPM.create_voxels(data, phases, array=im, path_min=path_min, csv=minerals_file, energy=energy, err=20, ps_cor=ps_cor, rows=2, cols=2, depth=2)
# # detected_phases = XRFPM.rejoin_vox_grid(vox, im_size) #doesnt work because image is too large
# sino = XRFPM.sinogram_rotation_3d(i0, Adet, Dsd, element, line, energy, vox, im_size, theta)
# XRFPM.display_ct_scans_3d(arrays=sino, opacity=0.1)

# recon = XRFPM.reconstruction_sinogram_3d(sino, theta)

# for i in range(len(recon)):
#     plt.imshow(recon[i])
#     plt.title(f"reconstruction for slice {i+1}")
#     plt.show()


"""
trial 9 -> regression between conc - sino - recon
"""
# energy = 90 #kev
# Adet = 30 #mm2
# Dsd = 30 #mm
# i0 = 1e10
# element = 'La'
# dtheta = 1
# line = 0
# theta = XRFPM.thetas(dtheta)
# ps_cor = 1 #µm

# concs = [0, 1, 10, 100, 500, 1000, 2000, 5000, 10000]
# it = []
# isino = []
# irecon = []

# multi_phases = [["Matrix", "1 ppm"], ["Matrix", "10 ppm"], ["Matrix", "100 ppm"], ["Matrix", "500 ppm"],
#                 ["Matrix", "1000 ppm"], ["Matrix", "2000 ppm"], ["Matrix", "5000 ppm"], ["Matrix", "10 000 ppm"]]

# for phase in multi_phases:
#     print(phase)
#     #create image
#     im_shape = 100, 100
#     im = np.full(im_shape, 100) #create matrix of La in 255 grayscale
#     im_size = im.shape
    
    
#     #display image
#     # XRFPM.display_ct_scan(array=im)
    
#     # #display histogram to find phases and grayscale values
#     # XRFPM.display_histogram(im)
#     # bkg_data, std_data = XRFPM.hist_to_phasedata(im, 4, 1) # doesnt work on clean data!!!
#     data = [0, 100]
    
#     #create pixels
#     pix = XRFPM.create_pixels(data, phase, array=im, path_min=path_min, csv=calibration_file, energy=energy, err=5, ps_cor=ps_cor, pixel_size_y = 1, pixel_size_x = 1, calculate_major=True)
#     detected_phases = XRFPM.rejoin_map_ct(pix, im_shape[1])
#     sino = XRFPM.sinogram_rotation_2d(i0, Adet, Dsd, element, line, energy, pix, im_size, theta)
#     recon = XRFPM.reconstruction_sinogram_2d(sino, theta)
    
#     indexmatrix = XRFPM.rows_index(pix, im_size)
#     pixelmatrix = XRFPM.rows_pixels(pix, im_size)
#     density, thickness, conc, mu_o, mu_f = XRFPM.preprocess_param(pix, indexmatrix, element, line, energy)
#     w_mu, fluor, abs_x, abs_y = XRFPM.FP(density, thickness, conc, mu_o, mu_f)
#     G = XRFPM.Geometryfactor(Adet, Dsd)
#     Qi = XRFPM.production_cross_section(element, line, energy)
#     ik = XRFPM.I_fluor(i0, G, Qi, w_mu, fluor, abs_x, abs_y)
    
#     it.append(ik.max())
#     isino.append(sino.max())
#     irecon.append(recon[0].max())
    
#     plt.imshow(sino)
#     plt.colorbar()
#     plt.show()
    
#     plt.imshow(recon[0].T, cmap='viridis')
#     plt.colorbar()
#     plt.show()


"""
trial - 10 -> Hayabusa sample A0186 (multiphase)
"""
# path="FILL IN WITH PATH TO .TIFF IMAGE"
# path_csv="FILL IN WITH PATH TO .CSV FILE"


# ps_cor = 1 #µm
# energy=90 #kev
# #trial_specific pixels
# img = XRFPM.ct_scan(path, "A0186-1_000449.tif")
# plt.imshow(img, cmap="gray")
# plt.colorbar()
# plt.show()

# imshape = img.shape

# # display histogram to find phases and grayscale values
# XRFPM.display_histogram(img)
# data = [7710,65000]
# #calculate mineral phases and pixels
# phases = ["Air","Magnetite"]


# pix = XRFPM.create_pixels(data, phases, path, "A0186-1_000449.tif", path_min=path_csv, csv=minerals_file, energy=90, err=20, ps_cor=ps_cor, pixel_size_x=1, pixel_size_y=1)
# detected_phases = XRFPM.rejoin_map_ct(pix, imshape[1])


# i0 = 1e10
# Adet = 30 #mm2
# Dsd = 120 #mm
# element = "Fe"
# line = 0 # ka1
# energy = 90 #kev
# im_size = img.shape
# dtheta = 1
# theta = XRFPM.thetas(dtheta)

# path5 = "/Users/Pedro/OneDrive - UGent/master 2/thesis/ct data/Pieter images/202301_id15_brenker/fit/"
# sinos_real = []
# recons_real = []

# with h5py.File(path5+"A0186_1_3_xrfct_scan1.h5", 'r') as f:
#     sinos = f["fit/channel00/ims"]
#     recons = f["tomo/channel00/ims"]
#     for i in range(len(sinos)):
#         sinos_real.append(sinos[i])
#         recons_real.append(recons[i])
        
# sinos_real = np.array(sinos_real)
# recons_real = np.array(recons_real)


# sino = XRFPM.sinogram_rotation_2d(i0, Adet, Dsd, element, line, energy, pix, im_size, theta)
# plt.imshow(sino)
# plt.ylabel(r"$\phi (degrees)$", fontsize=16)
# plt.xlabel(r"$Displacement (pixels)$", fontsize=16)
# plt.colorbar()
# plt.show()


# plt.imshow(sinos_real[1])
# plt.ylabel(r"$\phi (degrees)$", fontsize=16)
# plt.xlabel(r"$Displacement (pixels)$", fontsize=16)
# plt.colorbar()
# plt.show()

# recon = XRFPM.reconstruction_sinogram_2d(sino, theta)
# plt.imshow(recon[0].T)
# cbar = plt.colorbar()
# plt.ylabel(r"$Y (pixels)$", fontsize=16)
# plt.xlabel(r"$X (pixels)$", fontsize=16)
# plt.show()

# plt.imshow(recons_real[1])
# cbar = plt.colorbar()
# plt.ylabel(r"$Y (pixels)$", fontsize=16)
# plt.xlabel(r"$X (pixels)$", fontsize=16)
# plt.show()


"""
check 2d functions
"""
# indexmatrix = XRFPM.rows_index(pix, im_size)
# pixelmatrix = XRFPM.rows_pixels(pix, im_size)
# density, thickness, conc, mu_o, mu_f = XRFPM.preprocess_param(pix, indexmatrix, element, line, energy)
# w_mu, fluor, abs_x, abs_y = XRFPM.FP(density, thickness, conc, mu_o, mu_f)
# G = XRFPM.Geometryfactor(Adet, Dsd)
# Qi = XRFPM.production_cross_section(element, line, energy)
# ik = XRFPM.I_fluor(i0, G, Qi, w_mu, fluor, abs_x, abs_y)

# plt.imshow(density)
# plt.title(r'density ($g\,cm^{-3}$)', fontsize=20)
# plt.colorbar()
# plt.show()

# c = conc*1e6
# plt.imshow(c)
# plt.title(r'concentration (ppm)', fontsize=20)
# plt.colorbar()
# plt.show()

# plt.imshow(mu_o)
# plt.colorbar()
# plt.title(r'$µ_0$ ($cm^2\,g^{-1}$)', fontsize=20)
# plt.show()

# plt.imshow(mu_f)
# plt.title(r'$µ_1$ ($cm^2\,g^{-1}$)', fontsize=20)
# plt.colorbar()
# plt.show()

# plt.imshow(w_mu)
# plt.title(r'$\frac{w_{Fe, k}}{\mu_{k,0}}$', fontsize=20)
# plt.colorbar()

# plt.show()

# plt.imshow(fluor)
# plt.title(r'$fluo$', fontsize=20)
# plt.colorbar()
# plt.show()

# plt.imshow(abs_x)
# plt.title(r'$abs_{\Delta X}$', fontsize=20)
# plt.colorbar()
# plt.show()

# plt.imshow(abs_y)
# plt.title(r'$abs_{\Delta Y}$', fontsize=20)
# plt.colorbar()
# plt.show()

# plt.plot(ik)
# # plt.yscale("log")
# plt.show()


"""
check stuff 3d
"""

# indexmatrix = XRFPM.rows_index_3d(vox, im_size)
# pixelmatrix = XRFPM.rows_voxels(vox, im_size)
# density, thickness, conc, mu_o, mu_f = XRFPM.preprocess_param_3d(vox, indexmatrix, element, line, energy)
# w_mu, fluor, abs_x, abs_y = XRFPM.FP_3d(density, thickness, conc, mu_o, mu_f)
# G = XRFPM.Geometryfactor(Adet, Dsd)
# Qi = XRFPM.production_cross_section(element, line, energy)
# ik = XRFPM.I_fluor_conv(i0, G, Qi, w_mu, fluor, abs_x, abs_y)

# # print("Showing sample")
# # sample = np.sum(im, axis=0)
# # plt.imshow(sample)
# # plt.show()

# print("showing density")
# XRFPM.display_ct_scans_3d(arrays=density, screenshot=True, name=figuresaver+'Phantoms/3dsphere/density')
# # sample = np.sum(density, axis=0)
# # plt.imshow(sample)
# # plt.show()

# print("showing conc")
# XRFPM.display_ct_scans_3d(arrays=conc)
# # sample = np.sum(conc, axis=0)
# # plt.imshow(sample)
# # plt.show()

# print("showing mu_o ")
# XRFPM.display_ct_scans_3d(arrays=mu_o)
# # sample = np.sum(mu_o, axis=0)
# # plt.imshow(sample)
# # plt.show()

# print("showing mu_f")
# XRFPM.display_ct_scans_3d(arrays=mu_f)
# # sample = np.sum(mu_f, axis=0)
# # plt.imshow(sample)
# # plt.show()

# print("showing w_mu")
# XRFPM.display_ct_scans_3d(arrays=w_mu)
# # sample = np.sum(w_mu, axis=0)
# # plt.imshow(sample)
# # plt.show()

# print("showing fluor")
# XRFPM.display_ct_scans_3d(arrays=fluor)
# # sample = np.sum(fluor, axis=0)
# # plt.imshow(sample)
# # plt.show()

# print("showing abs_x")
# XRFPM.display_ct_scans_3d(arrays=abs_x)
# # sample = np.sum(abs_x, axis=0)
# # plt.imshow(sample)
# # plt.show()

# print("showing abs_y")
# XRFPM.display_ct_scans_3d(arrays=abs_y)
# sample = np.sum(abs_y, axis=1)
# plt.imshow(sample)
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.colors import LogNorm
# from matplotlib.ticker import MultipleLocator


# for i in range(len(indexmatrix)):
#     plt.imshow(im[i])
#     plt.show()
    
#     fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(18,7), constrained_layout=True)
#     fig.suptitle(f"calculated matrices for slice {i+1}")
    
#     ax1.set_title("density")
#     im1 = ax1.imshow(density[i])
#     divider1 = make_axes_locatable(ax1)
#     cax1 = divider1.append_axes("right", size="20%", pad=0.05)
#     cbar1 = plt.colorbar(im1, cax=cax1, format="%.2f")
    
#     ax2.set_title("conc")
#     im2 = ax2.imshow(conc[i])
#     divider2 = make_axes_locatable(ax2)
#     cax2 = divider2.append_axes("right", size="20%", pad=0.05)
#     cbar2 = plt.colorbar(im2, cax=cax2, format="%.2f")
    
#     ax3.set_title("mu_o")
#     im3 = ax3.imshow(mu_o[i])
#     divider3 = make_axes_locatable(ax3)
#     cax3 = divider3.append_axes("right", size="20%", pad=0.05)
#     cbar3 = plt.colorbar(im3, cax=cax3, format="%.2f")
    
#     ax4.set_title("mu_f")
#     im4 = ax4.imshow(mu_f[i])
#     divider4 = make_axes_locatable(ax4)
#     cax4 = divider4.append_axes("right", size="20%", pad=0.05)
#     cbar4 = plt.colorbar(im4, cax=cax4, format="%.2f")

#     ax5.set_title("w_mu")
#     im5 = ax5.imshow(w_mu[i])
#     divider5 = make_axes_locatable(ax5)
#     cax5 = divider5.append_axes("right", size="20%", pad=0.05)
#     cbar5 = plt.colorbar(im5, cax=cax5, format="%.2f")
    
#     ax6.set_title("fluor")
#     im6 = ax6.imshow(fluor[i])
#     divider6 = make_axes_locatable(ax6)
#     cax6 = divider6.append_axes("right", size="20%", pad=0.05)
#     cbar6 = plt.colorbar(im6, cax=cax6, format="%.2f")

#     ax7.set_title("abs_x")
#     im7 = ax7.imshow(abs_x[i])
#     divider7 = make_axes_locatable(ax7)
#     cax7 = divider7.append_axes("right", size="20%", pad=0.05)
#     cbar7 = plt.colorbar(im7, cax=cax7, format="%.2f")
    
#     ax8.set_title("abs_y")
#     im8 = ax8.imshow(abs_y[i])    
#     divider8 = make_axes_locatable(ax8)
#     cax8= divider8.append_axes("right", size="20%", pad=0.05)
#     cbar8 = plt.colorbar(im8, cax=cax8, format="%.2f")   
    
#     plt.show()

# plt.imshow(ik)
# plt.colorbar()
# plt.show()

