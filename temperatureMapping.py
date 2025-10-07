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
            print('Ok')