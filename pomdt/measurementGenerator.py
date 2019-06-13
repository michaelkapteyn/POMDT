from pomdt.utils import *

import numpy as np
import scipy as sp
import scipy.interpolate
from itertools import product
class measurementGenerator():
    def __init__(self, damageLibrary = None, noiseParams = None):
        self.dlib = damageLibrary
        self.noise = noiseParams if noiseParams is not None else noiseParams(self.dlib.nSensors)
        self.nMeasurements = damageLibrary.nSensors

    def getMeasurement(self, modeIdx, loadIdx, noisy = True, type = 'linear'):
        nSensors = self.dlib.nSensors
        nLoads = len(self.dlib.loadCases)
        # referenceMeasurements = np.zeros((nLoads,nSensors))
        # for i, load in enumerate(self.dlib.loadCases):
        #     referenceMeasurements[i,:] = np.array(list(self.dlib.predictedMeasurements[mode][load]))

        # Choose the interpolation type
        if type is 'linear':
            # fit = sp.interpolate.interp1d(np.arange(0, len(states)), referenceMeasurements, axis=0)
            # Create coordinate pairs
            lists = [range(0,len(self.dlib.states[self.dlib.components[0]])),range(0,len(self.dlib.states[self.dlib.components[1]])), range(0,len(self.dlib.loadCases))]
            coord = list(product(*lists))
            data  = [np.array(self.dlib.predictedMeasurements[mode][load]).transpose() for mode in self.dlib.modes for load in self.dlib.loadCases]

            # Interpolate
            interp = scipy.interpolate.LinearNDInterpolator(coord, data)
        else:
            print('Error: Unknown interpolation type:'+str(type))

        # Generate clean measurement
        cleanmeasurement = interp(modeIdx+(loadIdx,))

        if noisy:
            # Add artificial noise to measurement
            if self.noise.type is "Gaussian":
                noise = np.random.normal(self.noise.mean, self.noise.sigma,cleanmeasurement.shape)
                noisymeasurement = cleanmeasurement+noise
            else:
                noisymeasurement = cleanmeasurement
            return noisymeasurement
        else:
            return cleanmeasurement

class noiseParams():
    def __init__(self, nSensors, type = "Gaussian", sigma=0):
        self.type = type
        self.mean = 0
        self.sigma = sigma
