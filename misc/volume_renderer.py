import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import interpn
import os
import SimpleITK as sitk
import math

file_name = 'name.nii.gz'
"""
based on:
Create Your Own Volume Rendering (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
"""

class volume_renderer():
  def __init__(self):
    self.values  = []
    self.scalars = [] ### scalars=[{'r':1.0, 'g':0.0, 'b':0.0, 'a':0.8}]
    self.sigmas  = [] ### sigmas=[{'r':2.0, 'g':2.0, 'b':2.0, 'a':2.0}]
    
    self.ramps   = [{'v':0.5, 'w':0.1, 's':{'r':0.6, 'g':0.0, 'b':0.0, 'a':0.8}}]

  def add_value(self, value, scalars, sigmas={'r':0.2, 'g':0.2, 'b':0.2, 'a':0.2}):
    self.values.append(value)
    self.scalars.append(scalars)
    self.sigmas.append(sigmas)
    
  def add_ramp(self, value, scalars={'r':1.0, 'g':0.0, 'b':0.0, 'a':0.8}, width=0.02):
    self.ramps.append({'v': value, 'w':width, 's':scalars})
    
  def transferFunction(self, x):
    r = 0.0 
    g = 0.0
    b = 0.0
    a = 0.0
    for v, s, sig in zip(self.values, self.scalars, self.sigmas):
      r += s['r'] * np.exp( -((x - v)**2)/sig['r']**2) 
      g += s['g'] * np.exp( -((x - v)**2)/sig['g']**2) 
      b += s['b'] * np.exp( -((x - v)**2)/sig['b']**2) 
      a += s['a'] * np.exp( -((x - v)**2)/sig['a']**2) 

    return r,g,b,a

  def transferFunction_ramp(self, x, value=0.0, scalars={}, width=0.1):
    r = np.copy(x) 
    g = np.copy(x) 
    b = np.copy(x) 
    a = np.copy(x) 
          
    for c, i in zip([r,g,b,a],['r','g','b','a']):
      c[np.where(c < (value-width))] = 0.0
      c[np.where(c > value)]         = scalars[i]
      c[np.where(c < value)]        *= scalars[i] / width

    return r,g,b,a

  def render(self, vol=np.zeros((0,0)), angle=0.0, value=0.0):
    
    ### orient img
    datacube = np.flip(np.fliplr(np.flipud(vol)), 1)
    
    # Datacube Grid
    Nx, Ny, Nz = datacube.shape
    x = np.linspace(-Nx/2, Nx/2, Nx)
    y = np.linspace(-Ny/2, Ny/2, Ny)
    z = np.linspace(-Nz/2, Nz/2, Nz)
    points = (x, y, z)
    
    # Camera Grid / Query Points -- rotate camera view
    N           = 200
    c           = np.linspace(-N/2, N/2, N)
    qx, qy, qz  = np.meshgrid(c,c,c)
    qxR         = qx
    qyR         = qy * np.cos(angle) - qz * np.sin(angle) 
    qzR         = qy * np.sin(angle) + qz * np.cos(angle)
    qi          = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T
    
    # Interpolate onto Camera Grid
    camera_grid = interpn(points, datacube, qi, method='linear', bounds_error=False, fill_value=0.0).reshape((N,N,N))
    # matplotlib.image.imsave( 'MIP_' + str(i) + '.png',  np.max(camera_grid, axis=0))
    
    # Do Volume Rendering
    image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))

    for dataslice in camera_grid:
      r,g,b,a = self.transferFunction_ramp(dataslice, value, self.ramps[0]['s'], self.ramps[0]['w'])
      # r,g,b,a = self.transferFunction(dataslice)
      
      image[:,:,0] = a*r + (1-a)*image[:,:,0]
      image[:,:,1] = a*g + (1-a)*image[:,:,1]
      image[:,:,2] = a*b + (1-a)*image[:,:,2]
    
    # image = np.clip(image,0.0,1.0)
    image = image - np.min(image)
    image = image / np.max(image)
      
    #   # Plot Volume Rendering
    #   plt.figure(figsize=(4,4), dpi=80)
      
    #   plt.imshow(image)
    #   plt.axis('off')
      
    #   # Save figure
    #   plt.savefig('volumerender' + str(i) + '.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
    
    
    
    # # Plot Simple Projection -- for Comparison
    # plt.figure(figsize=(4,4), dpi=80)
    
    # plt.imshow(np.log(np.mean(datacube,0)), cmap = 'viridis')
    # plt.clim(-5, 5)
    # plt.axis('off')
    
    # # Save figure
    # plt.savefig('projection.png',dpi=300,  bbox_inches='tight', pad_inches = 0)
    # # plt.show()
  

    return image
