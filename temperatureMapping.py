import os,sys
sys.path.append(os.environ['HOME']+'/scripts')

from ThermalModel.utils.helpers import getPars
from ThermalModel.utils.shapes import readPlate

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

                vertices, triangles = readPlate(self.pars['shapeFile'])
                print(f"Dimensions of vertices: {sum(len(x) for x in vertices)}, triangles: {sum(len(x) for x in triangles)}")




            elif (self.pars['shapeFile'] == 'None') & (self.pars['shape_a'] != 'None'):
                print('Using an ellipsoid')
            else:
                raise ValueError('Shape file or ellipsoid missing, but required for temperature profile calculation!')
            
