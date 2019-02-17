import os
import numpy as np

from shmm.damageLibrary import *
from shmm.measurementGenerator import *
from shmm.shmm import *
from shmm.viz import *
from shmm.utils import *

DAMAGE_LIB_PATH = './damageLibrary.json'
np.set_printoptions(suppress=True)

# create damage library object
d = damageLibrary(DAMAGE_LIB_PATH)

# generate artificial measurements using linear interpolation through all reference states
nMeasurements = 200
m = measurementGenerator(d)
noise = noiseParams(d.nSensors,"Gaussian")
noisymeasurements, cleanmeasurements, groundTruthState = m.genMeasurements(nMeasurements, d.states, type='linear')
# Perform smoothing
model = SHMM(d, noisymeasurements, cleanmeasurements, groundTruthState, windowLength=20)
model.windowedSmoother()

# Create an animation showing smoothing results
animateWindowedSmoother(model)
