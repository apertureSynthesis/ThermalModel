import os,sys
sys.path.append(os.environ['HOME']+'/scripts')

from ThermalModel.utils.core import Layer, Surface
from ThermalModel.utils.helper import getPars

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.units as u

from astropy.modelng.physical_models import Blackbody
import time

class RadiativeTransfer(object):

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

            Optional:
            withPlots: whether to include optional plots
            displayHeader: whether to print a sample model header



    Outputs:
        model_gamma_I_emiss_J/tempMap_rt.fits: FITS file containing the surface emission maps for each
            combination of index of refraction and loss tangent
        model_plots.pdf: Summary plots of model contents
    """


    def __init__(self,parFile):

        self.parFile = parFile
        self.pars = getPars(parFile)

    def getRadiativeTransfer(self):

        #Check for the input file and output directories.
        if not os.path.exists(self.pars['tempMapPath']):
            raise ValueError('Temperature maps directory not found')
        
        if not os.path.exists(self.pars['radiativePath']):
            os.makedirs(self.pars['radiativePath'])

        #Define the ranges of index of refraction (nn) and loss tangent (loss_tan) to iterate over
        nn = np.linspace(self.pars['nn'][0], self.pars['nn'][1], self.pars['nn'][2])
        loss_tan = np.logspace(self.pars['loss'][0], self.pars['loss'][1], self.pars['loss'][2])

        #Find each of the map subdirectories
        mapDirs = [name for name in os.listdir(self.pars['tempMapPath'])
                   if os.path.isdir(os.path.join(self.pars['tempMapPath'], name))]
        
        #Make their mirror in the output folder
        for mapDir in mapDirs:
            if not os.path.exists(os.path.join(self.pars['tempMapPath'],mapDir)):
                os.makedirs(os.path.join(self.pars['tempMapPath'],mapDir))

        #Values to propagate into output maps
        keys = ['ti', 'emiss', 'rho', 'c', 'p_orb', 'p_rot', 'a_skin', 'd_skin', 'long']

        #Create the emission maps for each value of thermal inertia
        for mapDir in mapDirs:


