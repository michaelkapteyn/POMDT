import os
import numpy as np

from shmm.damageLibrary import *
from shmm.measurementGenerator import *
from shmm.shmm import *
from shmm.utils import *

DAMAGE_LIB_PATH = './damageLibrary.json'
np.set_printoptions(suppress=True)

# create damage library object
d = damageLibrary(DAMAGE_LIB_PATH)

# generate artificial measurements using linear interpolation through all reference states
nMeasurements = 100
m = measurementGenerator(d)
noise = noiseParams(d.nSensors,"Gaussian")
noisymeasurements, cleanmeasurements = m.genMeasurements(nMeasurements, d.states, type='linear')
# Perform smoothing
s = SHMM(d, noisymeasurements)
posterior = s.fwd_bkwd()
# print(posterior)

# Plot input measurements and smoothing results
s.plotmeasurements([0,5,10], cleanmeasurements)
s.plotsmoothingresult()
