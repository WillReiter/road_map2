#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import netCDF4 as nc
import matplotlib.colors as colors
os.environ['PROJ_LIB'] = r'/home/syao/anaconda3/envs/trajGRU/share/proj'
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from nowcasting.hmw_evaluation import pixel_to_dbz
from matplotlib.colors import LinearSegmentedColormap


def colormap():
	cdict = ['whitesmoke','dodgerblue','cyan','limegreen','green',\
							'yellow','goldenrod','orange','red','firebrick', 'darkred']
	return colors.ListedColormap(cdict)
	
def save_frame(data, save_path, normalization=False):
        
	if not normalization:
		data = pixel_to_dbz(data)
	#data = np.rot90(frame_dat.transpose(1,0))
	data[data<0] = 0
	data[data>60] = 60
	fig = plt.figure()
	ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	plt.rcParams["font.weight"] = "bold"
	plt.rcParams["axes.labelweight"] = "bold"
	# m = Basemap(projection='cyl', llcrnrlon=-99.5, llcrnrlat=26.5, urcrnrlon=-93.5, urcrnrlat=32.5, resolution='l')
	m = Basemap(projection='cyl', llcrnrlon=-125, llcrnrlat=36, urcrnrlon=-119, urcrnrlat=42, resolution='l')
	#cmap = LinearSegmentedColormap.from_list('cmap', ['whitesmoke','dodgerblue','cyan','limegreen','green',\
	#						'yellow','goldenrod','orange','red','firebrick', 'darkred'], 256)
	m.readshapefile(r"/home/syao/Desktop/trajGRU/gdam/gadm36_USA_1", 'US', drawbounds=True)
	lons, lats = m.makegrid(600, 600) #change here
	x, y = m(lons, lats)
	# parallels = np.arange(26.5, 32.51, 1)
	parallels = np.arange(36, 42.01, 1)
	m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)
	# meridians = np.arange(-99.5, -93.49, 1)
	meridians = np.arange(-125, -118.09, 1)
	m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)

	color = ['whitesmoke','dodgerblue','cyan','limegreen','green',\
							'yellow','goldenrod','orange','red','firebrick', 'darkred']
	clevs = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
	norm = colors.BoundaryNorm(clevs, 24)
	my_cmap = colors.ListedColormap(color)
	data[data <= 0] = None
	cs = m.contourf(x, y, data, clevs, colors=color)
	#cs = m.pcolormesh(x, y, data, latlon=True, vmin=0, vmax=60, cmap=my_cmap)
	cbar = m.colorbar(cs, location='right', pad="5%")
	cbar.set_label('Radar reflectivity (dBZ)', size=14)
	#plt.imshow(data, cmap=my_cmap, norm=norm, extent=(113, 120, 37, 42))
	#plt.imshow(data, cmap=my_cmap, extent=(113, 120, 37, 42))

	#plt.tight_layout()
	plt.savefig(save_path, dpi=900, bbox_inches='tight')
	plt.close()
	ax.clear()

if __name__ == "__main__":

        #frame_data = nc.Dataset("/media/4T/nc_700_500_2km_10min/20170508/000000.nc")['tbb_13'][:]
        #frame_data = frame_data - 273.15
        frame_data = nc.Dataset("/home/syao/Desktop/trajGRU/straitform_test/20190214/000035.nc")['DBZ'][:]
        frame_data = np.array(frame_data).squeeze()
        save_path = './'
        save_frame(frame_data[:,:], os.path.join(save_path, "101132.jpg"), normalization=True)






