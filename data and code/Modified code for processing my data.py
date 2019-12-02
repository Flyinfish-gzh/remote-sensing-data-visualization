#!/usr/bin/env python
# coding: utf-8

# In[3]:


import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
from matplotlib.colors import LogNorm


def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)
     
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    return heatmap.T, extent


fig, axs = plt.subplots(1, 2)

dataset = rasterio.open('E:/Jupyter Notebook/LC81490312016259LGN00/LC8_subset_layerstacking.tif')
red_band = dataset.read(4)
NIR_band = dataset.read(5)

np.seterr(divide='ignore', invalid='ignore')
ndvi = (NIR_band.astype(float)-red_band.astype(float))/(NIR_band.astype(float)+red_band.astype(float))

ndvi_flat = np.ndarray.flatten(ndvi)
red_band_flat = np.ndarray.flatten(red_band)

# Generate some test data
x = ndvi_flat
y = red_band_flat

sigmas = [0, 16]

for ax, s in zip(axs.flatten(), sigmas):
    if s == 0:
        ax.plot(x, y, 'k.', markersize=0.1)
        ax.set_title("Scatter plot")
        ax.set_xlabel('NDVI')
        ax.set_ylabel('Red Reflectance')
    else:
        img, extent = myplot(x, y, s)
        ax.imshow(img,extent=extent, origin='lower',cmap=cm.jet)
        ax.set_aspect('equal')
        
        ax.set_title("Smoothing with  $\sigma$ = %d" % s)
        ax.set_xlabel('NDVI')
        ax.set_ylabel('Red Reflectance')

plt.show()


# In[ ]:




