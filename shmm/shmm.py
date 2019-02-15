import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from shmm.damageLibrary import *
from shmm.measurementGenerator import *
from shmm.utils import *


class SHMM:
    def __init__(self, damageLibrary, measurements, initialProb=None, transitionMat=None, emissionMat=None):
        self.damageLibrary = damageLibrary
        self.states = self.damageLibrary.states
        self.measurements = measurements
        self.initialProb = initialProb if initialProb is not None else self.defaultInitialProb()
        self.transitionMat = transitionMat if transitionMat is not None else self.defaultTransitionMatrix()
        self.emissionMat = emissionMat if emissionMat is not None else self.defaultEmissionMatrix()


    def defaultInitialProb(self):
        initialProb = np.zeros((len(self.states),1))
        initialProb[0] = 1
        return initialProb

    def defaultTransitionMatrix(self):
        alpha = 0.07

        tp = np.zeros((1,len(self.states)))
        tp[0,0]= 1-alpha
        tp[0,1] = alpha

        transition_probability = tp
        for i in range(1,len(self.states)):
           transition_probability = np.vstack((transition_probability,np.roll(tp,i)))

        return transition_probability

    def defaultEmissionMatrix(self):
        sigma = 200
        E = np.ones((len(self.states),len(self.measurements)))
        for stIdx, st in enumerate(self.states):
            for mIdx, m in enumerate(self.measurements):
                for sensIdx in range(self.damageLibrary.nSensors):
                    E[stIdx,mIdx] = E[stIdx,mIdx]*norm.pdf(m[sensIdx], self.damageLibrary.predictedMeasurements[st][sensIdx], sigma)
        for mIdx, m in enumerate(self.measurements):
            E[:,mIdx] = E[:,mIdx]/np.linalg.norm(E[:,mIdx])
        return E

    def fwd(self,fv,ev):
        for s, state in enumerate(self.states):
            fv = np.multiply(self.emissionMat[:,ev],np.transpose(self.transitionMat).dot(fv).flatten())
            c = np.linalg.norm(fv)
        return fv.flatten()/c, c

    def bkwd(self,b,ev):
        for s, state in enumerate(self.states):
            b[s] = np.multiply(self.emissionMat[:,ev],b).dot(self.transitionMat[s,:])
        return b.flatten()

    def fwd_bkwd(self):
        print(self.emissionMat)
        nMeasurements = len(self.measurements)
        ev = list(range(0,nMeasurements))

        states = self.states
        start_prob = self.initialProb

        # forward part of the algorithm
        fwd = np.zeros((nMeasurements,len(states)))
        c = np.zeros(nMeasurements)
        for i, observation_i in enumerate(ev):
            if i == 0:
                fwd[i,:],c[i] = self.fwd(start_prob,ev[i])
            else:
                fwd[i,:],c[i] = self.fwd(fwd[i-1,:],ev[i])

        # backward part of the algorithm
        sv = np.zeros((nMeasurements,len(states)))
        b = np.ones(len(states))
        for i, observation_i in reversed(list(enumerate(ev))):
            sv[i,:] = np.multiply(fwd[i,:],c[i]*b)
            sv[i,:] = sv[i,:]/np.linalg.norm(sv[i,:])
            b = self.bkwd(b,ev[i])

        self.fullposterior = sv
        return sv

    def plotsmoothingresult(self):
        # plot smoothing result
        plt.plot(range(len(self.measurements)), np.argmax(self.fullposterior,1),label="Smoothing Result")

        # plot ground truth
        plt.plot(range(len(self.measurements)), np.linspace(0,len(self.states)-1,len(self.measurements)),label = "Ground Truth")
        plt.show()

    def plotmeasurements(self, indices, cleanmeasurements=None):
        measurementcolors = ['red','green','blue']
        for plotIdx, sensorIdx in enumerate(indices):
            if cleanmeasurements is not None:
                plt.plot(range(len(self.measurements)), cleanmeasurements[:,sensorIdx],color='black')
            plt.scatter(range(len(self.measurements)), self.measurements[:,sensorIdx],color=measurementcolors[plotIdx])
        plt.show()
