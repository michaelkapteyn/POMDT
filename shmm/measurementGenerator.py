from shmm.utils import *

import numpy as np
import scipy as sp
import scipy.interpolate
class measurementGenerator():
    def __init__(self, damageLibrary = None):
        self.dlib = damageLibrary
        self.noise = noiseParams(self.dlib.nSensors)

    def genMeasurements(self, nMeasurements = 200, states = None, type = 'linear'):
        # Handle defaults
        if states is None:
            states = self.dlib.states

        # Get reference measurements from damage library
        nSensors = self.dlib.nSensors
        referenceMeasurements = np.zeros((len(states),nSensors))
        for i,state in enumerate(states):
            referenceMeasurements[i,:] = np.array(list(self.dlib.predictedMeasurements[state]))

        # Choose the interpolation type
        if type is 'linear':
            fit = sp.interpolate.interp1d(np.arange(0, len(states)), referenceMeasurements, axis=0)
        else:
            print('Error: Unknown interpolation type:'+str(type))

        # Generate clean measurements
        measurements = fit(np.linspace(0, len(states)-1, nMeasurements))

        # Add artificial noise to measurements
        if self.noise.type is "Gaussian":
            noise = np.random.normal(self.noise.mean, self.noise.sigma,measurements.shape)
            noisymeasurements = measurements+noise
        else:
            noisymeasurements = measurements

        return noisymeasurements, measurements

class noiseParams():
    def __init__(self, nSensors, type = "Gaussian"):
        self.type = type
        self.mean = 0
        self.sigma = 200
