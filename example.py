import os
import numpy as np

from pomdt import *


DAMAGE_LIB_PATH = './damageLibrary.json'
np.set_printoptions(suppress=True)

# create damage library object
d = DamageLibrary(DAMAGE_LIB_PATH)


# create measurement generator to generate artificial measurements using linear interpolation through reference measurements (later replace with RB-FEA call)
noise = noiseParams(d.nSensors,"Gaussian",sigma=50)
m = measurementGenerator(d, noise)


nTimeSteps = 2
groundTruthState = list(zip(np.linspace(0,len(d.states[d.components[0]])-1,nTimeSteps),np.linspace(0,len(d.states[d.components[0]])-1,nTimeSteps)))
pomdp = POMDP(nTimeSteps, d.modes, d.loadCases, d.capability, m, groundTruthState, transitionMat = d.TPM)

fig = plt.figure(figsize=(10, 8))
sensor_fig = plt.figure(figsize=(10, 8))

viz = Visualizer(fig, sensor_fig, pomdp,d.states)

assimilator = Assimilator(pomdp, 1)

sim = Simulator(pomdp, assimilator, viz)
sim.run()
