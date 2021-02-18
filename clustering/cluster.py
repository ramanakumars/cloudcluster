import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import os
from matplotlib.colors import rgb_to_hsv, LogNorm
from matplotlib.gridspec import GridSpec
import netCDF4 as nc

class ImageCluster():
    '''
        The ImageCluster class handles the clustering of 
        different cloud types of a map projected JunoCam image
    '''

    def __init__(self, projfile):
        '''
            Initializer method

            Parameters
            ----------
            projfile : string
                name of the netCDF4 file from projecting JunoCam image. 
                See the `projection` module
        '''
        self.projfile   = projfile
        self.fname, ext = os.path.splitext(projfile)

        indata = nc.Dataset(projfile, 'r')

        self.image = indata.variables['img_corr'][:].astype(float)/255.

        self.lon = indata.variables['lon'][:].astype(np.float64)
        self.lat = indata.variables['lat'][:].astype(np.float64)

        self.extents = (self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max())

        LON, LAT = np.meshgrid(self.lon, self.lat)

        self.latflat = LAT.flatten()
        self.lonflat = LON.flatten()

        ## correct JPG images
        if(ext[1:] == 'jpg'):
            self.image = self.image/255.

        self.ny, self.nx, _ = self.image.shape

        self.get_data_from_img()

        fig, ax = plt.subplots(1,1, figsize=(11, 8))
        ax.imshow(self.image, extent=self.extents, origin='lower')
        plt.show()

    def get_data_from_img(self):
        '''
            Construct the 1-D arrays that hold the RGB and HSV
            data from the image
        '''
        self.hsvimage  = rgb_to_hsv(self.image)

        ## find the pixels that correspond to data 
        self.pixmask = \
            np.where(np.min(self.image, axis=2).flatten() > 0.)[0]
        
        self.npix           = len(self.pixmask)

        self.hsvdata = np.zeros((self.npix, 3))
        self.rgbdata = np.zeros((self.npix, 3))

        for i in range(3):
            hsvi =  self.hsvimage[:,:,i].flatten()
            rgbi =  self.image[:,:,i].flatten()
            self.hsvdata[:,i] = hsvi[self.pixmask]
            self.rgbdata[:,i] = rgbi[self.pixmask]


    def create_img_from_array(self, data):
        ''' 
            Convert between the flattened array and an image
            
            Parameters
            ----------
            data : numpy.ndarray
                Input 1D flattened array. Should be the same length 
                as `self.pixmask`

            Outputs
            -------
            img : numpy.ndarray
                Output 2D array with values from `data` filled in the appropriate
                pixels and zeros elsewhere

            Raises
            ------
            AssertionError
                if the data size does not match the `self.pixmask` size
        '''

        assert len(data) == len(self.pixmask), "input data not the same size as pixmask!"

        if(len(data.shape) > 1):
            img = np.empty((self.ny, self.nx, 3))
        else:
            img = np.empty((self.ny, self.nx))

        img[:] = np.nan

        for i in range(self.npix):
            pixi = self.pixmask[i]
            jj = pixi//self.nx
            ii = pixi - jj*self.nx

            img[jj,ii] = data[i]
        return img

    def create_clusters(self, n_clusters=4, axis=(0,1), source='hsv', **kwargs):
        '''
            Main driver for the classifier. Uses KMeans clustering to classify
            the input data. Can be RGB or HSV and can use different components
            of each.


            Parameters
            ----------
            n_clusters : int
                The number of clusters to classify [Default: 4]
            source : string
                Use either the RGB ('rgb') or HSV ('hsv') values [Defualt: hsv]
            axis : tuple
                The two components to use for the clustering algorithm
            **kwargs : 
                Other arguments to be used in the KMeans class

            Raises
            ------
            ValueError
                if source is not either 'rgb' or 'hsv'

        '''
        self.clustdata     = np.zeros((self.npix,2))

        if(source=='hsv'):
            self.clustdata[:,0] = self.hsvdata[:,axis[0]]
            self.clustdata[:,1] = self.hsvdata[:,axis[1]]
        elif(source=='rgb'):
            self.clustdata[:,0] = self.rgbdata[:,axis[0]]
            self.clustdata[:,1] = self.rgbdata[:,axis[1]]
        else:
            raise ValueError("source argument must be rgb or hsv")

        clustering = KMeans(n_clusters=n_clusters, **kwargs).fit(self.clustdata)
        centers    = clustering.cluster_centers_
        labels     = clustering.labels_

        ## sort the labels numbers increasing in x-value
        oldlabs    = np.argsort(centers[:,0])
        newlabs    = labels.copy()
        self.centers = centers[oldlabs,:].copy()

        ## rename the labels so that the first one is label 1
        for li, labeli in enumerate(oldlabs):
            newlabs[labels==labeli] = li + 1

        self.labels      = newlabs.copy()
        self.nlabels     = np.unique(labels).shape[0]
    
    def plot_clusters(self):
        '''
            Plots the original image, the labeled data and a 2D histogram
        '''
        ## create a colormap to plot the labels
        cmap        = plt.cm.cubehelix(np.linspace(0, 1, self.nlabels+1))

        ## create a image with the labels so that we can compare
        labelimg    = np.zeros(self.npix)
        labelimg[:] = self.labels

        labelimg    = self.create_img_from_array(labelimg)

        fig = plt.figure(figsize=(10,20))
        ax1 = fig.add_subplot('311')
        ax2 = fig.add_subplot('312')
        ax3 = fig.add_subplot('313')
        plt.subplots_adjust(top = 0.92, left=0.0, right=0.98, bottom=0.12, hspace=0.13, wspace=0.10)

        ax1.imshow(self.image, extent=self.extents, origin='lower')
        ax2.imshow(labelimg, vmin=0, vmax=self.nlabels, origin='lower',\
                   cmap='cubehelix', extent=self.extents)

        for axi in [ax1, ax2]:
            axi.set_xticks([])
            axi.set_yticks([])

        ax1.set_title(r'Raw image')
        ax2.set_title(r'Cloud classes')

        ax3.cla()
        h = ax3.hist2d(np.degrees(self.clustdata[:,0]), self.clustdata[:,1], bins=40, cmap='RdPu', norm=LogNorm())
        #fig.colorbar(h[3], ax=ax3)
        for ii in range(self.nlabels):
            ax3.scatter(np.degrees(self.centers[ii,0]), self.centers[ii,1], s=60, marker='o', c=cmap[ii+1], edgecolor='#bebebe')

        ax3.set_xlabel(r"Hue [$\degree$]")
        ax3.set_ylabel(r"Saturation")

        ax3.set_title(r"Histogram of pixel values")

        plt.show()

    def filter_cluster(self, label):
        '''
            Remove a specific label from the cluster

            Parameters
            ----------
            label : int
                The number of the label to remove
        '''
        label_mask = np.where(self.labels==label)[0]

        self.pixmask = np.delete(self.pixmask, label_mask, axis=0)
        self.rgbdata = np.delete(self.rgbdata, label_mask, axis=0)
        self.hsvdata = np.delete(self.hsvdata, label_mask, axis=0)

        print("Removed %d pixels"%(len(label_mask)))

        self.npix = len(self.pixmask)
    
    def cluster_stats(self):
        '''
            Plots the latitudinal distribution of clusters
        '''
        lon = self.lonflat[self.pixmask]
        lat = self.latflat[self.pixmask]

        cmap        = plt.cm.cubehelix(np.linspace(0, 1, self.nlabels+1))

        fig, ax = plt.subplots(figsize=(10,10))

        latunique, nlats = np.unique(lat, return_counts=True)
        
        data  = np.zeros((self.lat.size, self.nlabels))

        for i, lati in enumerate(self.lat):
            nlati  = nlats[latunique==lati]
            latsub = lat[lat==lati]
            if(nlati > 0.):
                for j in range(self.nlabels):
                    data[i,j] = len(lat[self.labels==j+1])/nlati

        for j in range(self.nlabels):
            ax.plot(data[:,j], self.lat, '-', color=cmap[j+1])

        ax.set_ylabel(r'Latitude [deg]')
        ax.set_xlabel(r'Relative number of labeled pixels')

        plt.show()
