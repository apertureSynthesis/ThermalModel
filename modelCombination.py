import os, sys
sys.path.append(os.environ['HOME']+'/scripts')

from ThermalModel.utils.helpers import getPars

from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.units as u

class modelCombination(object):

    """
    Class for combining all radiative transfer models for a given frequency into a single FITS file.
    Optionally plot out fluxes from those models.
    Output files can be used for model fitting by user in an external step.
    """

    def __init__(self, parFile):
        super().__init__()

        self.parFile = parFile
        self.pars = getPars(parFile)

    def getCombinedModel(self):

        #Check if we are making a combined model
        if self.pars['doCombinedModel']:
            #Check for input and output directories
            if not os.path.exists(self.pars['radiativePath']):
                raise ValueError('Radiative transfer maps not found')
            
            if not os.path.exists(self.pars['combinedModelPath']):
                os.makedirs(self.pars['combinedModelPath'])

            #Find each radiative transfer model in their various subdirectories
            modelDirs = [name for name in os.listdir(self.pars['radiativePath'])
                         if os.path.isdir(os.path.join(self.pars['radiativePath'], name))]
            
            if len(modelDirs) == 0:
                raise ValueError("No radiative transfer models found")
            
            #List of longitudes to consider
            lonList = np.asarray(self.pars['subEarthLongitudes'], dtype=np.float64)

            #Organize the files, prepare to read them in, and work out the parameter space
            records = []
            for modelDir in modelDirs:
                modelBase = os.path.basename(modelDir)
                gamma = float(modelBase.split('_')[2])
                emis = float(modelBase.split('_')[-1])
                records.append((gamma, emis, modelDir))

            #Sort the unique values of thermal inertia and emissivity
            uniqueGamma = sorted({gamma for gamma, emis, modelDir in records})
            uniqueEmis = sorted({emis for gamma, emis, modelDir in records})

            #Map to an index for a stable sorting(first thermal inertia, then emissivity)
            gIndex = {g: i for i, g in enumerate(uniqueGamma)}
            eIndex = {e: i for i, e in enumerate(uniqueEmis)}

            #Sort the folders
            records.sort(key = lambda t: (gIndex[t[0]], eIndex[t[1]]))

            nTI = len(uniqueGamma)
            nEmis = len(uniqueEmis)
            if not self.pars['suppressMessages']:
                print(f"Found {nTI:0d} unique TI's and {nEmis:0d} unique emissivities.")

            #Prepare to integrate the models and combine
            pxlScl = u.Quantity(self.pars['pxlScale'])
            pxlSclOmega = (pxlScl.to("rad"))**2
            pxlSclOmegaSr = pxlSclOmega.to("sr")

            #Shape nti*nemis, nlon, nref, nloss
            fluxAll = []
            #Corresponding brightness temperature
            tbAll = []
            #projected area per image (sr), shape nti*nemis, nlon
            aAll = []
            #one TI per (TI, emis) pair
            gammaOut = []
            #one emis per (TI, emis) pair
            emisOut = []
            nRef = nLoss = None

            for gamma, emis, modelDir in records:
                modelFiles = sorted(glob(os.path.join(self.pars['radiativePath'],modelDir, '*_rt_???.fits')))
                if not modelFiles:
                    raise RuntimeError(f"No FITS files matched in {modelDir}")
                
                nLon = len(modelFiles)

                fluxD = []
                tbD = []
                aD = []

                for modelFile in modelFiles:
                    with fits.open(modelFile) as mF:
                        image = mF[0].data
                        bunit = u.Unit(mF[0].header.get('bunit', 'Jy / sr'))
                        image *= bunit #surface brightness units

                        #Read reference arrays
                        if 'REFIDX' in mF:
                            nnL = np.array(mF['REFIDX'].data)
                        elif 'refidx' in mf:
                            nnL = np.array(mF['refidx'].data)
                        else:
                            raise KeyError(f"'REFIDX' not found in {modelFile}")
                        
                        if 'LOSS' in mF:
                            lossL = np.array(mF['LOSS'].data)
                        elif 'loss' in mF:
                            lossL = np.array(mF['loss'].data)
                        else:
                            raise KeyError(f"'LOSS' not found in {modelFile}")
                        
                    #Confirm shape: nref, nloss, ny, nx
                    if image.ndim != 4:
                        raise ValueError(f"Expected an image with 4 dimensions (nref, nloss, ny, nx), found {image.ndim} in {modelFile}")
                    
                    if nRef is None:
                        nRef = image.shape[0]
                        nLoss = image.shape[1]
                    else:
                        if (nRef, nLoss) != (image.shape[0], image.shape[1]):
                            raise ValueError(f"Inconsistent (nRef, nLoss): {image.shape[:2]} vs ({nRef},{nLoss}) in {modelFile}")
                            
                    #Integrate the flux density over the pixels:
                    #sum(I_nu [Jy/sr]) * dOmega[sr] -> [Jy]
                    #sum over x,y
                    #Shape nRef, nLoss
                    flux = (image.sum(axis=-1).sum(axis=-1) * pxlSclOmegaSr).to(u.Jy)

                    #Projected area = number of on-disk pixels * dOmega; on-disk is >0
                    onDisk = image[0, 0] > 0
                    nPix = int(onDisk.astype(int).sum())
                    areaSr = (nPix * pxlSclOmegaSr).to(u.sr)

                    #Average brightness temperature
                    #Treat the integrated flux as if from a "beam" of area = projected area
                    #so flux/beam = average over the disk
                    #Use astropy equivalency: Jy/Beam <-> K at freq, beam area
                    avgFluxPerBeam = flux #Jy per beam (whole disk area)
                    tb = avgFluxPerBeam.to(
                        u.K,
                        equivalencies = u.brightness_temperature(u.Quantity(self.pars['freq']), areaSr)
                    ) #Shape nRef, nLoss

                    fluxD.append(flux)
                    tbD.append(tb)
                    aD.append(areaSr)

                #Build longitude list from file if necessary
                if len(self.pars['subEarthLongitudes']) == 0:
                    lonList = list(range(0, 360, 90))[:nLon]

                #Stack along longitudinal axis -> nLon, nRef, nLoss
                fluxD = u.Quantity(fluxD) #nLon, nRef, nLoss
                tbD = u.Quantity(tbD) #nLon, nRef, nLoss
                aD = u.Quantity(aD) #nLon

                fluxAll.append(fluxD)
                tbAll.append(tbD)
                aAll.append(aD)
                gammaOut.append(gamma)
                emisOut.append(emis)

            #Convert the lists to arrays
            fluxAll = u.Quantity(fluxAll) #nTI*nEmis, nLon, nRef, nLoss
            tbAll = u.Quantity(tbAll) #nTI*nEmis, nLon, nRef, nLoss
            aAll = u.Quantity(aAll) #nTI*nEmis, nLon
            gammaOut = u.Quantity(gammaOut) #nTI*nEmis
            emisOut = u.Quantity(emisOut) #nTI*nEmis

            #Reshape to (nTI, nEmis, nLon, nRef, nLoss)
            try:
                fluxAll = fluxAll.reshape(nTI, nEmis, fluxAll.shape[1], nRef, nLoss)
                tbAll = tbAll.reshape(nTI, nEmis, tbAll.shape[1], nRef, nLoss)
                aAll = aAll.reshape(nTI, nEmis, aAll.shape[1])
                gamma2D = gammaOut.reshape(nTI, nEmis)
                emis2D = emisOut.reshape(nTI, nEmis)
            except:
                raise RuntimeError(f"Reshape failed, check folder counts and parsing")
        
            #Defensively sort by increasing TI
            tiSort = np.argsort(gamma2D[:, 0])
            fluxAll = fluxAll[tiSort]
            tbAll = tbAll[tiSort]
            aAll = aAll[tiSort]
            gammaSort = gamma2D[tiSort][:, 0] #nTI
            emisSort = emis2D[tiSort][0, :] #nEmis

            #Cross-section (projected area) averaged over emissivity (nLon, nTI)
            aReshape = aAll.mean(axis=1).T #nLon, nTI

            #name output file
            outFile = os.path.join(self.pars['combinedModelPath'],'combinedModel.fits')
            # ---------------
            # Build FITS HDUs
            # ---------------
            hdu0 = fits.PrimaryHDU(tbAll.value)     # data: K
            hdu0.header['BUNIT']   = 'K'
            hdu0.header['AX1_NAME'] = 'loss'    # axis order: (ti, emis, lon, refidx, loss) in our arrays?
            hdu0.header['AX2_NAME'] = 'refidx'  # NOTE: Our data array order is (ti, emis, lon, ref, loss)
            hdu0.header['AX3_NAME'] = 'lon'
            hdu0.header['AX4_NAME'] = 'emis'
            hdu0.header['AX5_NAME'] = 'ti'
            # To be extra clear, record the exact shape:
            hdu0.header['SHAPE0'] = tbAll.shape[0]  # nti
            hdu0.header['SHAPE1'] = tbAll.shape[1]  # nemis
            hdu0.header['SHAPE2'] = tbAll.shape[2]  # nlon
            hdu0.header['SHAPE3'] = tbAll.shape[3]  # nref
            hdu0.header['SHAPE4'] = tbAll.shape[4]  # nloss

            hdu1 = fits.ImageHDU(fluxAll.to(u.Jy).value, name='FLUX')
            hdu1.header['BUNIT'] = 'Jy'

            # Use the last-read nn/loss; they should be consistent across files/folders
            hdu2 = fits.ImageHDU(lossL.astype(np.float64), name='LOSS')
            hdu3 = fits.ImageHDU(nnL.astype(np.float64),   name='REFIDX')

            hdu4 = fits.ImageHDU(np.asarray(lonList, dtype=np.float64), name='LON')
            hdu4.header['BUNIT'] = 'deg'

            hdu5 = fits.ImageHDU(np.asarray(emisSort, dtype=np.float64), name='EMIS')

            hdu6 = fits.ImageHDU(np.asarray(gammaSort, dtype=np.float64), name='TI')
            hdu6.header['BUNIT'] = 'tiu'

            hdu7 = fits.ImageHDU(aReshape.to(u.sr).value, name='XSEC')
            hdu7.header['BUNIT'] = 'sr'
            hdu7.header['DESC']  = 'Projected area per (lon,ti) averaged over emissivity'

            fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7]).writeto(outFile, overwrite=True)

            if not self.pars['suppressMessages']:
                print(f"Saved: {outFile}") 

    def plotCombinedModel(self):
        
        if self.pars['doModelPlotting']:
            #Open the model
            modelFile = os.path.join(self.pars['combinedModelPath'],'combinedModel.fits')

            with fits.open(modelFile) as mF:
                flux = mF['FLUX'].data
                lonVals = mF['LON'].data
                gamma = mF['ti'].data
                loss = mF['loss'].data
                refIdx = mF['refidx'].data
                emis = mF['emis'].data

            #Plot a lightcurve with fixed e, n, and loss tangent, with varying TI
            fig, ax = plt.subplots(figsize=(7,5))
            for i in range(len(gamma)):
                ax.plot(lonVals, flux[i, 2, :, 2, -2], 'o')
            plt.text(0.8, 0.1, f"$\\epsilon$ = {emis[2]:.2f}", transform=ax.transAxes)
            plt.text(0.8, 0.06, f"n = {refIdx[i]:.2f}", transform=ax.transAxes)
            plt.text(0.8, 0.02, f"$\\tan\\Delta$ = {loss[-2]:.2f}", transform=ax.transAxes)
            ax.set_xlabel('Longitude (deg)')
            ax.set_ylabel('Flux (Jy)')
            plt.legend(gamma)

            plt.tight_layout()
            plt.savefig(self.pars['combinedModelPath']+'/combinedModel.pdf',dpi=300)            


    def __call__(self):
        self.getCombinedModel()
        self.plotCombinedModel()