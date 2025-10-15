import os,sys
sys.path.append(os.environ['HOME']+'/scripts')

from ThermalModel.utils.helpers import getPars

import numpy as np

class temperatureMapping(object):

    def __init__(self, parFile):
        super().__init__()

        self.parFile = parFile
        self.pars = getPars(parFile)

    def getTemperatureMaps(self):

        if self.pars['doTemperatureMapping'] == 'True':

            #Check if a shape model file is provided. If so, use it.
            #If not, use the tri-axial ellipsoid
            if self.pars['shapeFile'] != 'None':
                print('Using a shape model: any ellipsoid will be ignored.')
            elif (self.pars['shapeFile'] == 'None') & (self.pars['shape_a'] != 'None'):
                print('Using an ellipsoid')
            else:
                raise ValueError('Shape file or ellipsoid missing, but required for temperature profile calculation!')