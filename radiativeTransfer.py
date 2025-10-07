import os,sys
sys.path.append(os.environ['HOME']+'/scripts')

from ThermalModel.utils.core import Layer, Surface
from ThermalModel.utils.helpers import getPars

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.units as u

from astropy.modeling.physical_models import BlackBody

from glob import glob

class radiativeTransfer(object):

    """
    Class for applying radiative transfer models to a set of simulated thermal maps for an asteroid.

    This class converts subsurface temperature profiles into surface emission fluxes (in units Jy/arcsec**2)
    at a given frequency. It iterates over a grid of dielectric properties (index of refraction and loss tangent)
    defined by the user. Outputs are saved in FITS files.

    Inputs:
        parFile: parameter (text) file containing fixed physical parameters and paths. Must contain:
            Physical parameters:
            object name
            asteroid rotational period (hours)
            asteroid orbital period (years)
            asteroid regolith density (kg m^-3)
            asteroid regolith specific heat (J K^-1 kg^-1)

            Modeling parameters:
            observing frequency
            range of index of refraction to test [start, stop, inc] to be fed to np.linspace
            range of loss tangent to test [start, stop, inc] to be fed to np.logspace

            Paths:
            krcFile: path to KRC model file
            reCastPath: path to store recast models
            tempMapPath: path where temperature models are stored
            radiativePath: path to save radiative transfer models

            Steps:
            doPreProcessing: Should the pre-processing step be performed?
            doRadiativeTransfer: Should the radiative transfer step be performed?
            *Note: it is possible to generate plots for existing files without running these steps

            Optional:
            withPlots: whether to include optional plots
            displayHeader: whether to print a sample model header



    Outputs:
        model_gamma_I_emiss_J/tempMap_rt.fits: FITS file containing the surface emission maps for each
            combination of index of refraction and loss tangent
        model_plots.pdf: Summary plots of model contents
    """


    def __init__(self,parFile):
        super().__init__()

        self.parFile = parFile
        self.pars = getPars(parFile)

    def getRadiativeTransfer(self):
        #Check that we are doing the radiative transfer and not just plotting
        if self.pars['doRadiativeTransfer'] == 'True':
            #Check for the input file and output directories.
            if not os.path.exists(self.pars['tempMapPath']):
                raise ValueError('Temperature maps directory not found')
            
            if not os.path.exists(self.pars['radiativePath']):
                os.makedirs(self.pars['radiativePath'])

            #Define the ranges of index of refraction (nn) and loss tangent (loss_tan) to iterate over
            nn = np.linspace(self.pars['nn'][0], self.pars['nn'][1], np.int64(self.pars['nn'][2]))
            loss_tan = np.logspace(self.pars['loss'][0], self.pars['loss'][1], np.int64(self.pars['loss'][2]))

            #Find each of the map subdirectories
            mapDirs = [name for name in os.listdir(self.pars['tempMapPath'])
                    if os.path.isdir(os.path.join(self.pars['tempMapPath'], name))]
            
            #Make their mirror in the output folder
            for mapDir in mapDirs:
                if not os.path.exists(os.path.join(self.pars['radiativePath'],mapDir)):
                    os.makedirs(os.path.join(self.pars['radiativePath'],mapDir))

            #Values to propagate into output maps
            keys = ['ti', 'emiss', 'rho', 'c', 'p_orb', 'p_rot', 'a_skin', 'd_skin', 'long']

            #Create the emission maps for each value of thermal inertia
            for mapDir in mapDirs:
                #Find the maps for each longitude value
                imFiles = glob(os.path.join(self.pars['tempMapPath'],mapDir) + '/tempmap_???.fits')

                #Loop through each image
                for imFile in imFiles:
                    print(f"Processing {os.path.basename(imFile)}")

                    #Load the model
                    with fits.open(imFile) as imFits:
                        intensity = BlackBody(imFits[0].data * u.K)(u.Quantity(self.pars['freq'])).to_value('Jy/arcsec2')
                        zz = imFits[1].data
                        emi = imFits[2].data
                        hdr = imFits[0].header

                    #Loop through the dielectric parameter space
                    images = np.zeros(nn.shape + loss_tan.shape + intensity.shape[1:])
                    for i in range(intensity.shape[-2]):
                        for j in range(intensity.shape[-1]):
                            if intensity[0, i, j] <= 0:
                                continue
                            #Initiate a layer with fixed n and loss tangent
                            layer = Layer(n=1.5, loss_tangent=1e-2, profile=[zz, intensity[:, i, j]])
                            surface = Surface(layer)

                            for k, n in enumerate(nn):
                                for t, l in enumerate(loss_tan):
                                    surface.layers[0].n = n
                                    surface.layers[0].loss_tangent = l
                                    images[k, t, i, j] = surface.emission(emi[i, j], u.Quantity(self.pars['freq']).to_value('m', u.spectral()))

                    #Save the simulated images
                    tmp = os.path.basename(imFile).split('_')
                    outFile = '_'.join(tmp[:-1] + ['rt'] + [tmp[-1]])
                    hdu = fits.PrimaryHDU(images.astype('float32'))
                    hdu.header['bunit'] = 'Jy/arcsec2'

                    for key in keys:
                        hdu.header[key] = hdr[key], hdr.comments[key]

                    hdu1 = fits.ImageHDU(nn.astype('float32'), name='refidx')
                    hdu2 = fits.ImageHDU(loss_tan.astype('float32'), name='loss')

                    fits.HDUList([hdu, hdu1, hdu2]).writeto(os.path.join(self.pars['radiativePath'],mapDir,outFile))

    def makePlots(self):
        
        if self.pars['plotRadiativeTransfer'] == 'True':
            #Find each of the output model subdirectories
            radDirs = [name for name in os.listdir(self.pars['radiativePath'])
                    if os.path.isdir(os.path.join(self.pars['radiativePath'], name))]
            
            #Plot a sample file
            radFiles = glob(os.path.join(self.pars['radiativePath'],radDirs[0])+'/*.fits')

            if not radFiles:
                print(f"No radiative transfer models found at {self.pars['radiativePath']}")
            else:
                with fits.open(radFiles[0]) as rF:
                    images = rF[0].data
                    hdr = rF[0].header
                    nn = rF[1].data
                    loss_tan = rF[2].data

                plotImage = images[0,0,:,:]*u.Unit(hdr['bunit']).to(u.K, equivalencies=u.brightness_temperature(u.Quantity(self.pars['freq'])))
                fig, ax = plt.subplots(figsize=(6,6), dpi=200)
                plotIm = ax.imshow(plotImage,origin='lower',cmap='inferno')
                cbar = plt.colorbar(plotIm)
                cbar.set_label('Brightness Temperature (K)')
                plt.title(f"Sample Brightness Plot for {self.pars['object']}")
                plt.tight_layout()
                plt.savefig(os.getcwd()+'/radiativeModel.pdf')


    def __call__(self):
        self.getRadiativeTransfer()
        self.makePlots()

