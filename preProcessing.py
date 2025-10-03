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

    def getPars(self):
        """
        Read in content from the input parameter file 
        and store them in a dictionary for later use
        """

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

    def reCast(self):


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

                outFile = self.pars['reCastPath']+f'/temp_gamma_{t.value:.0f}_emiss_{e:.2f}.fits'
                fits.HDUList([hdu0, hdu1, hdu2, hdu3]).writeto(outFile, overwrite=True)

    def makePlots(self):
        """
        Make sample plots and display headers if requested
        """

        if self.displayHeader:
            print('KRC File Header Information:\n')
            print(self.pars['krcFile'].info())

        if self.withPlots:

            """
            Plot of depths sampled by each thermal inertia value
            """
            #Load the depth data
            with fits.open(self.pars['krcFile']) as f:
                TIs = np.squeeze(f[3].data)    #Thermal Inertia
                emis  = np.squeeze(f[4].data)  #Emissivity
                depths = np.squeeze(f[5].data) #Depths
                

            fig = plt.figure(figsize=(15,15), dpi=200)

            for i in range(depths.shape[1]):
                plt.plot(np.arange(depths.shape[0]),depths[:,i],label=f"TI={TIs[i]:.0f}")
            plt.ylabel('Depth (m)')
            plt.xlabel('Index')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.getcwd()+'/TI-Depth.pdf')
            plt.show()

            """
            Plots of (1) Surface temperature vs. LST: how surface temperature varies with TI through a diurnal cycle
            (2) Temperature vs. Depth: how temperature propagates into the subsurface at different times
            """
            tfiles = [fits.open(f) for f in [self.pars['reCastPath']+f'/temp_gamma_{t:.0f}_emiss_0.90.fits' for t in TIs]]

            fig, axs = plt.subplots(figsize=(15,30), dpi=200)
            axs = axs.ravel()
            for tfile in tfiles:
                axs[0].plot(tfile[0].data[9, 0]) #latitude index 9 (near equator), depth = 0 (surface)
                axs[0].set_xlabel('LST index (per 0.5 hours from noon)')
                axs[0].set_ylabel('T (K)')
                axs[0].legend([f'TI={ti:.0f}' for ti in TIs])

            for hour in range(0, 48, 8):
                axs[1].plot(tfile[0].data[9, :, hour])
                axs[1].set_xlabel('Depth (Index)')
                axs[1].set_ylabel('T (K)')
                axs[1].legend([f'TI={ti:.0f}' for ti in TIs])

            plt.tight_layout()
            plt.savefig(os.getcwd()+'/Temperature-Plots.pdf')
            plt.show()

            """
            Plot of temperature vs. time and depth for the first combination of TI and emissivity
            """
            modelFile = self.pars['reCastPath']+f'/temp_gamma_{TIs[0]:.0f}_emiss_{emis[0]:.2f}.fits'
            with fits.open(modelFile) as f:
                temp  = f[0].data  #Temperature
                lst   = f['lst'].data #Local solar times (48 points in hours)
                depth = f['depth'].data #Depths (100 points, in meters)
                lat   = f['lat.data'] #Latitudes (19 pionts, in degrees)

            #Select a latitude to visualize (index 9 = equator)
            ilat = 9
            z = temp[ilat, :, :].T #Slice and transpose to shape (time, depth)

            #Create a meshgrid for plotting time vs. depth
            #The meshgrid must match the shape of z: (48 times steps x 100 depth points)
            LST, DEPTH = np.meshgrid(lst, depth, indexing='ij')

            #Create the 3D plot
            fig, ax = plt.subplots(figsize=(15,15), projection='3d', dpi=200)
            
            surf = ax.plot_surface(LST, DEPTH, z, cmap='inferno')
            ax.set_xlabel('Local Solar Time (hours)')
            ax.set_ylabel('Depth (m)')
            ax.set_zlabel('Temperature (K)')
            ax.set_title(f"Temperature vs. Time and Depth\nTI={TIs[0]:.0f}, Emissivity={emis[0]:.2f}, Latitude={lat[ilat]:.0f} deg")

            #Add colorbar
            fig.colorbar(surf, label='Temperature (K)')

            plt.tight_layout()
            plt.savefig(os.getcwd()+'/Temperature-Depth-Time.pdf')
            plt.show()

            """
            Plot of Temperature vs LST and Depth (2D)
            """
            

            

    def __call__(self):
        self.getPars()
        self.reCast()




