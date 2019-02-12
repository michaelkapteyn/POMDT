import os
import numpy as np


from shmm.damageLibrary import *
from shmm.measurementGenerator import *
from shmm.shmm import *
from shmm.utils import *

DAMAGE_LIB_PATH = './damageLibrary.json'

# create damage library object
d = damageLibrary(DAMAGE_LIB_PATH)

# generate artificial measurements - 100 measurements using linear interpolation through all states
nMeasurements = 200
m = measurementGenerator(d)
noise = noiseParams(d.nSensors,"Gaussian")
meas = m.genMeasurements(nMeasurements, d.states, type='linear')

print(meas)
# Perform smoothing
#TODO
