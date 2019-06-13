import os
import numpy as np

from pomdt.pomdp import *
from pomdt.simulator import *
from pomdt.assimilator import *
from pomdt.damageLibrary import *
from pomdt.measurementGenerator import *

from pomdt.viz import *
from pomdt.utils import *


DAMAGE_LIB_PATH = './damageLibrary.json'
np.set_printoptions(suppress=True)

# create damage library object
d = damageLibrary(DAMAGE_LIB_PATH)



# create measurement generator to generate artificial measurements using linear interpolation through reference measurements (later replace with RB-FEA call)
noise = noiseParams(d.nSensors,"Gaussian",sigma=50)
m = measurementGenerator(d, noise)


# Perform smoothing
nTimeSteps = 200
groundTruthState = list(zip(np.linspace(0,len(d.states[d.components[0]])-1,nTimeSteps),np.linspace(0,len(d.states[d.components[0]])-1,nTimeSteps)))
pomdp = POMDP(nTimeSteps, d.modes, d.loadCases, m, groundTruthState, transitionMat = d.TPM)
viz = Visualizer(pomdp,d.states)
viz.initializePlotWindow()
sim = Simulator(pomdp, viz)
sim.run()

# model.windowedSmoother()
# model.fullSmoother()

# Create an animation showing smoothing results
# viz.animateWindowedSmoother()
