import os
import numpy as np


from shmm.damageLibrary import *
from shmm.measurementGenerator import *
from shmm.shmm import *
from shmm.utils import *

DAMAGE_LIB_PATH = './damageLibrary.json'

# # create damage library object
# d = damageLibrary(DAMAGE_LIB_PATH)
#
# # generate artificial measurements - 100 measurements using linear interpolation through all states
# nMeasurements = 200
# m = measurementGenerator(d)
# noise = noiseParams(d.nSensors,"Gaussian")
# meas = m.genMeasurements(nMeasurements, d.states, type='linear')
#
# print(meas)
# Perform smoothing
#TODO

dl = read_json_file('./damageLibrary.json')

states = list(dl.keys())[:-1]
end_state = states[-1]

observations = np.array(([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]))

start_probability = np.zeros(len(states))
start_probability[0] = 1

tp = np.array(([0.9, 0.1, 0, 0, 0]))
transition_probability = tp
for i in range(1,len(states)):
   transition_probability = np.vstack((transition_probability,np.roll(tp,i)))



ep = np.array(([0.5, 0.25, 0, 0, 0.25]))
emission_probability = ep
for i in range(1,len(states)):
   emission_probability = np.vstack((emission_probability,np.roll(ep,i)))


fwd,bkw,pos = fwd_bkw(observations, states, start_probability, transition_probability, emission_probability, end_state)
print(pos)
