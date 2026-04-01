import os,sys
sys.path.append(os.environ['HOME']+'/scripts')

from ThermalModel.utils.helpers import getPars
from ThermalModel.utils.shapes import readPlate, meshEllipsoid, meshNormal, vectMatchView, rd2xyz, meshGeoMap, tpmSph2Plt, tpmMapping, readObjPlate

from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.units as u

from astropy.modeling.physical_models import BlackBody

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

        if self.pars['doTemperatureMapping']:

            #Check if the specified output path exists
            if not os.path.exists(self.pars['tempMapPath']):
                os.makedirs(self.pars['tempMapPath'])

            #Check if a shape model file is provided. If so, use it.
            #If not, use the tri-axial ellipsoid
            if self.pars['shapeFile'] != 'None':
                print('Using a shape model: any ellipsoid will be ignored.')

                #Detect whether we have a .obj or .plt file
                if self.pars['shapeFile'].lower().endswith('.plt'):
                    vertices, triangles = readPlate(self.pars['shapeFile'])
                elif self.pars['shapeFile'].lower().endswith('.obj'):
                    vertices, triangles = readObjPlate(self.pars['shapeFile'])
                else:
                    raise ValueError('Shape file must have .obj or .plt extension.')


            elif (self.pars['shapeFile'] == 'None') & (self.pars['shape_a'] != 'None'):
                print('Using an ellipsoid')

                vertices, triangles = meshEllipsoid(np.float64(self.pars['a']), np.float64(self.pars['b']), 
                                                    np.float64(self.pars['c']), np.float64(self.pars['lat_step']), 
                                                    np.float64(self.pars['lon_step']))
            else:
                raise ValueError('Shape file or ellipsoid missing, but required for temperature profile calculation!')
            
            print('---------------\n')

            #Location of the recast KRC models
            tFiles = glob(self.pars['reCastPath']+'/*tref_gamma*_emis*.fits')

            #FITS keys to propogate to output
            keys = ['ti', 'emiss', 'rho', 'cs', 'p_orb', 'p_rot', 'a_skin', 'd_skin']

            #Prepare longitudes
            subeLon = self.pars['subEarthLongitudes']
            nLon = len(subeLon)
            subeLat = np.zeros(nLon) + np.float64(self.pars['subEarthLatitude'])
            subsLon = (subeLon + np.float64(self.pars['deltaLon'])) % 360
            subsLat = np.zeros(nLon) + np.float64(self.pars['subSolarLatitude'])
            rH = np.zeros(nLon) + np.float64(self.pars['rH'])
            delta = np.zeros(nLon) + np.float64(self.pars['delta'])

            #Iterate through each model
            for tFile in tFiles:
                if not self.pars['suppressMessages']:
                    print(f"Processing file: {tFile}")

                #Load reference temperature array
                with fits.open(tFile) as tf:
                    tref  = tf[0].data  #Temperature
                    thdr  = tf[0].header
                    lst   = tf['lst'].data #Local solar times (48 points in hours)
                    lsthdr  = tf['lst'].header
                    hourAngle = ((lst - 12) * 15 + 360) % 360 #hour angle from local noon in degrees
                    depth = tf['depth'].data #Depths (100 points, in meters)
                    dhdr  = tf['depth'].header
                    lat   = tf['lat'].data #Latitudes (19 points, in degrees)
                    lathdr= tf['lat'].header
                #Reshape tref to match IDL expectation
                tref = np.transpose(tref, axes=[2,1,0])

                #Prepare output directory
                outStem = tFile.split('/')[-1][:-5]
                tOutDir = os.path.join(self.pars['tempMapPath'],outStem)
                if not os.path.exists(tOutDir):
                    os.makedirs(tOutDir)


                #Loop through the longitudes
                for i in range(len(subsLon)):

                    if not self.pars['suppressMessages']:
                        print(f"Sub-Earth longitude: {subeLon[i]}")

                    sunPos = vectMatchView(rd2xyz([subsLon[i], subsLat[i]]), subeLat[i], subeLon[i], 0) * rH[i] * 1.496e8
                    newVertices = vectMatchView(vertices, subeLat[i], subeLon[i], 0)
                    res = np.float64(self.pars['pxlScale']) * delta[i] * 1.496e8 / 206265000 #Image resolution in km from mas

                    #Note - although shadow-casting is available in the code, it has historically not been implemented.
                    iMap, eMap, aMap, mask, pltMap = meshGeoMap(newVertices, triangles, sunPos, delta[i]*1.496e8, xres=res, yres=res,
                                                                xs = int(self.pars['xSize']), ys=int(self.pars['ySize']), noShadow=True)

                    #Save the plate map and plate files
                    suffix = '_' + f"{np.round(subeLon[i]):03d}.fits"
                    outFile = tOutDir + '/platemap' + suffix

                    hdu0 = fits.PrimaryHDU(pltMap)
                    hdr0 = hdu0.header
                    hdr0['long'] = (subeLon[i], 'sub-Earth longitude (deg)')

                    hdu1 = fits.ImageHDU(data = eMap, header = hdr0)
                    hdu2 = fits.ImageHDU(data = iMap, header = hdr0)
                    hdu3 = fits.ImageHDU(data = aMap, header = hdr0)

                    fits.HDUList([hdu0,hdu1,hdu2,hdu3]).writeto(outFile, overwrite=True)
                    if not self.pars['suppressMessages']:
                        print(f"Wrote {outFile}\n-----------------\n")

                    #Calculate TPM for plate shape model
                    tempList = tpmSph2Plt(vertices, triangles, tref, subsLon[i], times = np.append(hourAngle,360), lats=lat)

                    #Temperature image
                    tempMap = tpmMapping(tempList, pltMap, depth=depth, zz=depth)

                    #Save simulation images
                    outFile = tOutDir + '/tempmap' + suffix
                    hdu0 = fits.PrimaryHDU(tempMap)
                    hdr0 = hdu0.header
                    hdr0['bunit'] = 'K'
                    for k in range(len(keys)):
                        val = thdr[keys[k]]
                        hdr0[keys[k]] = val
                    hdr0['long'] = (subeLon[i], 'sub-Earth longitude (deg)')
                    hdu1 = fits.ImageHDU(data = depth, header = hdr0)
                    hdu2 = fits.ImageHDU(data = eMap, header = hdr0)
                    fits.HDUList([hdu0,hdu1,hdu2]).writeto(outFile, overwrite=True)

    def makePlots(self):

        if self.pars['plotTemperatureMapping']:
            #Find each of the output model subdirectories
            #Find each of the map subdirectories
            mapDirs = [name for name in os.listdir(self.pars['tempMapPath'])
                    if os.path.isdir(os.path.join(self.pars['tempMapPath'], name))]
            if len(mapDirs) == 0:
                raise ValueError("No temperature maps found")
            

            #Find the maps for each longitude value for the first set of regolith parameters
            imFiles = sorted(glob(os.path.join(self.pars['tempMapPath'],mapDirs[0]) + '/platemap_*.fits'))
            tempFiles = sorted(glob(os.path.join(self.pars['tempMapPath'],mapDirs[0]) + '/tempmap_*.fits'))
            outputDir = os.path.join(self.pars['tempMapPath'],mapDirs[0])
            fig, axs = plt.subplots(len(imFiles),4,figsize=(9,5*len(imFiles)))
            #Loop through each image
            imInd = 0
            for imFile, tempFile, lon in zip(imFiles, tempFiles, self.pars['subEarthLongitudes']):
                if not self.pars['suppressMessages']:
                    print(f"Processing {os.path.basename(imFile)}")

                #Load the model
                with fits.open(imFile) as imFits:
                    pltMap = imFits[0].data
                    eMap = imFits[1].data
                    iMap = imFits[2].data
                    aMap = imFits[3].header

                with fits.open(tempFile) as tFits:
                    intensity = BlackBody(tFits[0].data * u.K)(u.Quantity(self.pars['freq'])).to_value('Jy/arcsec2')

                im0 = axs[imInd,0].imshow(intensity[0,...],origin='lower',cmap='viridis')
                cbar0 = plt.colorbar(im0, ax=axs[imInd,0],fraction=0.046,pad=0.04,location='top')
                cbar0.set_label(f'Intensity for \nsubEarth longitude {lon:.0f} deg')

                im1 = axs[imInd,1].imshow(pltMap,origin='lower',cmap='viridis')
                cbar1 = plt.colorbar(im1, ax=axs[imInd,1],fraction=0.046,pad=0.04,location='top')
                cbar1.set_label('Plate Map Index')                

                im2 = axs[imInd,2].imshow(iMap,origin='lower',cmap='viridis')
                cbar2 = plt.colorbar(im2, ax=axs[imInd,2],fraction=0.046,pad=0.04,location='top')
                cbar2.set_label('Incidence Angle')

                im3 = axs[imInd,3].imshow(eMap,origin='lower',cmap='viridis')
                cbar3 = plt.colorbar(im3, ax=axs[imInd,3],fraction=0.046,pad=0.04,location='top')
                cbar3.set_label('Emission Angle')


                imInd += 1

            plt.tight_layout()
            plt.savefig(outputDir+'/tempMaps.pdf',dpi=300)

    def __call__(self):
        self.getTemperatureMaps()
        self.makePlots()

                    


