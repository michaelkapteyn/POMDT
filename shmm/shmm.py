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
        initialProb = np.zeros(len(self.states))
        initialProb[0] = 1
        return initialProb

    def defaultTransitionMatrix(self):
        alpha = 0.1

        tp = np.zeros((1,len(self.states)))
        tp[0,0]= 1-alpha
        tp[0,1] = alpha

        transition_probability = tp
        for i in range(1,len(self.states)):
           transition_probability = np.vstack((transition_probability,np.roll(tp,i)))
        return transition_probability

    def defaultEmissionMatrix(self):
        sigma = 0.4
        E = np.ones((len(self.states),len(self.measurements)))
        for stIdx, st in enumerate(self.states):
            for mIdx, m in enumerate(self.measurements):
                for sensIdx in range(self.damageLibrary.nSensors):
                    E[stIdx,mIdx] = E[stIdx,mIdx]*norm.pdf(m[sensIdx], self.damageLibrary.predictedMeasurements[st][sensIdx], sigma)
        return E

    def fwd_bkw(self):
        observations = list(range(0,len(self.measurements)))
        states = self.states[:-1]
        end_st = len(self.states)-1

        start_prob = self.initialProb
        trans_prob = self.transmissionMatrix
        emm_prob = self.emmissionMatrix

        # forward part of the algorithm
        fwd = []
        # f_prev = np.zeros((len(states),1))
        for i, observation_i in enumerate(observations):
            f_curr = np.zeros((len(states),1))
            for st,p in enumerate(states):
                if i == 0:
                    # base case for the forward part
                    prev_f_sum = start_prob[st]
                else:
                    prev_f_sum = sum(f_prev[k]*trans_prob[k,st] for k,s in enumerate(states))

                f_curr[st] = emm_prob[st,observation_i] * prev_f_sum

            fwd.append(f_curr)
            f_prev = f_curr

        p_fwd = sum(f_curr[k] * trans_prob[k][end_st] for k,s in enumerate(states))

        # backward part of the algorithm
        bkw = []
        b_prev = np.zeros((len(states),1))
        for i, observation_i_plus in enumerate(reversed(np.append(observations[1:],0))):
            b_curr = np.zeros((len(states),1))
            for st,s in enumerate(states):
                if i == 0:
                    # base case for backward part
                    b_curr[st] = trans_prob[st][end_st]
                else:
                    b_curr[st] = sum(trans_prob[st][l] * emm_prob[l][observation_i_plus] * b_prev[l] for l,s in enumerate(states))

            bkw.insert(0,b_curr)
            b_prev = b_curr

        p_bkw = sum(start_prob[l] * emm_prob[l][observations[0]] * b_curr[l] for l,s in enumerate(states))

        # merging the two parts
        posterior = np.zeros((len(observations),len(states)))
        for i in range(len(observations)):
            for st,s in enumerate(states):
                posterior[i,st] = fwd[i][st] * bkw[i][st] / p_fwd
            # posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st,s in enumerate(states)})
        assert p_fwd-p_bkw < 1e-5
        # return fwd, bkw, posterior
        self.fullposterior = posterior
        return posterior

    def plotsmoothingresult(self):
        # plot smoothing result
        plt.plot(range(len(self.measurements)), np.argmax(self.fullposterior,1),label="Smoothing Result")

        # plot ground truth
        plt.plot(range(len(self.measurements)), np.linspace(0,len(self.states)-1,len(self.measurements)),label = "Ground Truth")
        plt.show()
