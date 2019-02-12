import numpy as np

from shmm.damageLibrary import *
from shmm.measurementGenerator import *
from shmm.utils import *


class SHMM:
    def __init__(self, damageLibrary, measurements, transitionProbabilities):
        self.initialprob = np.zeros(len(damage.library.states))
        self.initialprob[0] = 1
        self.transitionmat = self.computeTransitionMatrix()
        self.emissionmat = self.computeEmissionMatrix()

    def computeTransitionMatrix(self):
        #TODO
        return transitionMat

    def computeEmissionMatrix(self):
        #TODO
        return emissionMat
