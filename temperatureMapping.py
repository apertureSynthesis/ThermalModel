import os,sys
sys.path.append(os.environ['HOME']+'/scripts')

from ThermalModel.utils.helpers import getPars
from ThermalModel.utils.shapes import readPlate, meshEllipsoid, vectMatchView, rd2xyz

from glob import glob

import numpy as np

from astropy.io import fits

"""
To-do:
50x50 image grid
make asteroid cover 70% of the image based on its size
get shape a,b,c from mesh lab
"""

class temperatureMapping(object):

    def __init__(self, parFile):
        super().__init__()

        self.parFile = parFile
        self.pars = getPars(parFile)

    def getTemperatureMaps(self):

        if self.pars['doTemperatureMapping'] == 'True':

            #Check if the specified output path exists
            if not os.path.exists(self.pars['tempMapPath']):
                os.makedirs(self.pars['tempMapPath'])

            #Check if a shape model file is provided. If so, use it.
            #If not, use the tri-axial ellipsoid
            if self.pars['shapeFile'] != 'None':
                print('Using a shape model: any ellipsoid will be ignored.')

                vertices, triangles = readPlate(self.pars['shapeFile'])
                print(f"Dimensions of vertices: {sum(len(x) for x in vertices)}, triangles: {sum(len(x) for x in triangles)}")


            elif (self.pars['shapeFile'] == 'None') & (self.pars['shape_a'] != 'None'):
                print('Using an ellipsoid')

                vertices, triangles = meshEllipsoid(self.pars['a'], self.pars['b'], self.pars['c'], self.pars['lat_step'], self.pars['lon_step'])
            else:
                raise ValueError('Shape file or ellipsoid missing, but required for temperature profile calculation!')
            
            #Location of the recast KRC models
            tFiles = glob(self.pars['reCastPath']+'/*tref_gamma*_emis*.fits')

            #FITS keys to propogate to output
            keys = ['ti', 'emiss', 'rho', 'c', 'p_orb', 'p_rot', 'a_skin', 'd_skin']

            #Prepare longitudes
            nLon = 360 / self.pars['deltaLon']
            subeLon = np.arange(nLon) / nLon * 360
            subeLat = np.zeros(nLon) + self.pars['subEarthLatitude']
            subsLon = (subeLon + self.pars['subEarthLongitude']) % 360
            subsLat = np.zeros(nLon) + self.pars['subSolarLatitude']
            rH = np.zeros(nLon) + self.pars['rH']
            delta = np.zeros(nLon) + self.pars['delta']

            #Iterate through each model
            for tFile in tFiles:
                print(f"Processing file: {tFile}")

                #Load reference temperature array
                with fits.open(tFile) as tf:
                    temp  = tf[0].data  #Temperature
                    thdr  = tf[0].header
                    lst   = tf['lst'].data #Local solar times (48 points in hours)
                    lsthdr  = tf['lst'].header
                    depth = tf['depth'].data #Depths (100 points, in meters)
                    dhdr  = tf['depth'].header
                    lat   = tf['lat'].data #Latitudes (19 pionts, in degrees)
                    lathdr= tf['lat'].header

                #Prepare output directory
                outStem = tFile.split('.')[0]
                tOutDir = os.path.join(self.pars['tempMapPath'],outStem)
                if not os.path.exists(tOutDir):
                    os.makedirs(tOutDir)

                #Loop through the longitudes
                for i in range(len(subsLon)):

                    print(f"Sub-Earth longitude: {subeLon[i]}")

                    #Calculate observing geometry
                    sunPos = vectMatchView(rd2xyz([subsLon[i], subsLat[i]]), subeLat[i], subeLon[i], 0) * rH[i] * 1.496e8
                    newVertices = vectMatchView(vertices, subeLat[i], subeLon[i], 0)
                    res = self.pars['pxlScale'] * delta[i] * 1.496e8 / 206265000 #Image resolution in km from mas

                    


