import os,sys
sys.path.append(os.environ['HOME']+'/scripts')

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import ascii
from astropy.io import fits
import astropy.units as u

class preProcessing(object):

    """
    Class for pre-processing KRC models for mapping onto an asteroid shape model.
    KRC models are hypercubes with dimensions of: 
    Local Time x Latitude x Depth x Thermal Inertia x Emissivity.
    Mapping the KRC model to the shape model requires recasting to a 
    cube of dimensions (N, D, M):
    N local solar times starting from local noon, D depths, and M latitudes.

    This class breaks each temperature model into individual files containing
    one combination of thermal inertia and emissivity. Optionally, plots are generated

    Inputs:
        parFile: parameter (text) file containing fixed physical parameters and paths. Must contain:
            Physical parameters:
            object name
            asteroid rotational period (hours)
            asteroid orbital period (years)
            asteroid regolith density (kg m^-3)
            asteroid regolith specific heat (J K^-1 kg^-1)

            Paths:
            krcFile: path to KRC model file
            reCastPath: path to store recast models

            Optional:
            withPlots: whether to include optional plots
            displayHeader: whether to print a sample model header



    Outputs:
        model_gamma_I_emiss_J.fits: FITS file containing the re-cast model for combination
            (I,J) of thermal inertia and emissivity
        model_plots.pdf: Summary plots of model contents
    """

    def __init__(self, parFile):
        super().__init__()

        self.parFile = parFile

    def _getPars(self):
        #Read in content from the input parameter file 
        #and store them in a dictionary for later use

        self.pars = {}
        with open(self.parFile, 'r') as f:
            for line in f:
                if not line.startswith("#"):
                    if line.rstrip():
                        split_line = line.rstrip().split("#")
                        var, val = split_line[0].split('=')
                        var = var.replace('"','')
                        val = val.replace('"','')
                        self.pars[var.strip()] = val.strip()

    def _reCast(self):


        #Check for the input file and output directories.
        if not os.path.exists(self.pars['krcFile']):
            raise ValueError('KRC input file not found')
        
        if not os.path.exists(self.pars['reCastPath']):
            os.makedirs(self.pars['reCastPath'])

        """
        Open the file.
        
        Extension 0 contains temperature data with indices
        [Depth, Emissivity, Thermal Inertia, Latitude, Local Solar Time]

        Extension 1 contains only the Local Solar Time
        Extension 2 contains only the Latitude
        Extension 3 contains only the Thermal Inertia
        Extension 4 contains only the Emissivity
        Extension 5 contains only the Depth
        """
        with fits.open(self.pars['krcFile']) as f:
            temp  = f[0].data * u.K      #Temperature
            lst   = f[1].data * u.hour   #Local Solar Time
            lat   = np.squeeze(f[2].data) * u.deg  #Latitude
            ti    = np.squeeze(f[3].data) * u.J / u.m**2 / u.K / (u.s**(1/2))  #Thermal Inertia
            emis  = np.squeeze(f[4].data)  #Emissivity
            depth = np.squeeze(f[5].data) * u.m  #Depth

        #Reorganize Extension 0 so that Local Solar Time comes first
        temp = np.moveaxis(temp, 0, 3)
        
        #Change start time to noon
        startTime = np.where(lst == 12 * u.hour)[0]
        lst = np.roll(lst, -startTime)

        #Check that the time was successfully shifted
        if lst[0] != 12 * u.hour:
            raise ValueError("Incorrect time rephasing")
        
        #Now shift the temperature values
        temp = np.roll(temp, -startTime, axis=-1)

        #Generate the files, one for each combination of Thermal Inertia and Emissivity
        for i, e in enumerate(emis):
            for j, t in enumerate(ti):
                #Slice the temperature model
                tref = temp[i, j]

                #Save the reference temperature model in Extension 0
                hdu0 = fits.PrimaryHDU(tref.to_value('K'))
                hdu0.header['bunit'] = 'K'
                hdu0.header['ti'] = t.to_value('J/(m2 K s(1/2))'), 'Thermal Inertia'
                hdu0.header['emiss'] = e, 'Emissivity'
                hdu0.header['rho'] = u.Quantity(self.pars['rho']).to_value('kg/m3'), 'Density [kg/m**3]'
                hdu0.header['cs'] = u.Quantity(self.pars['cs']).to_value('J/(K kg)'), 'Specific heat [J/(K kg)]'
                hdu0.header['p_orb'] = u.Quantity(self.pars['orbital_period']).to_value('year'), 'Orbital Period [year]'
                hdu0.header['p_rot'] = u.Quantity(self.pars['rotational_period']).to_value('hour'), 'Rotational Period [hour]'

                hdu0.header['a_skin'] = (np.sqrt(u.Quantity(self.pars['orbital_period']) / np.pi) * t / (u.Quantity(self.pars['rho']) * u.Quantity(self.pars['cs']))).to_value('m'), 'Annual Thermal Skin Depth [m]'
                hdu0.header['a_skin'] = (np.sqrt(u.Quantity(self.pars['rotational_period']) / np.pi) * t / (u.Quantity(self.pars['rho']) * u.Quantity(self.pars['cs']))).to_value('m'), 'Diurnal Thermal Skin Depth [m]'

                #Extension 1 for Local Solar Time
                hdu1 = fits.ImageHDU(lst.to_value('hour'), name='lst')
                hdu1.header['bunit'] = 'hour'

                #Extension 2 for Depths
                hdu2 = fits.ImageHDU(depth[:, j].to_value('m'), name='depth')
                hdu2.header['bunit'] = 'm'

                #Extension 3 for latitude
                hdu3 = fits.ImageHDU(lat.to_value('deg'), name='lat')
                hdu3.header['bunit'] = 'deg'

                outFile = self.pars['reCastPath']+'/temp_gamma_{:.0f}_emiss_{:.2f}.fits'.format(t.value, e)
                fits.HDUList([hdu0, hdu1, hdu2, hdu3]).writeto(outFile, overwrite=True)


        
    def __call__(self):
        self._getPars()
        self._reCast()




