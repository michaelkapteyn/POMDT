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
nMeasurements = 100
m = measurementGenerator(d)
noise = noiseParams(d.nSensors,"Gaussian")
measurements = m.genMeasurements(nMeasurements, d.states, type='linear')

# Perform smoothing
s = SHMM(d, measurements)
posterior = s.fwd_bkw()

s.plotsmoothingresult()
