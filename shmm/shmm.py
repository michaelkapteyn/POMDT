import numpy as np

from shmm.damageLibrary import *
from shmm.measurementGenerator import *
from shmm.utils import *
from queue import *
from functools import reduce

class SHMM:
    def __init__(self, damageLibrary, measurements, initialProb=None, transitionMat=None, emissionMat=None):
        self.damageLibrary = damageLibrary
        self.measurements = measurements

        if initialProb is None:
            self.initialprob = np.zeros(len(damage.library.states))
            self.initialprob[0] = 1
        else:
            self.initialProb = np.array(initialProb)


        if transitionMat is None:
            self.TM = self.computeTransitionMatrix()
        else:
            self.TM = transitionMat

        if emissionMat is None:
            self.OM = self.computeEmissionMatrix()
        else:
            self.OM = emissionMat

        self.delay = 20

        self.events = Queue()
        self.t = 0
        self.B = np.ndarray(self.TM.shape)
        self.forward = np.dot(self.initialProb, self.TM)
        self.backward = np.dot(self.initialProb, self.TM)

    def computeTransitionMatrix(self):
        T = []
        return T

    def computeEmissionMatrix(self):
        O =[]
        return O

def fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st):
    end_st = 4
    # forward part of the algorithm
    fwd = []
    f_prev = {}
    for i, observation_i in enumerate(observations):
        f_curr = np.zeros((len(states)))
        for st,p in enumerate(states):
            if i == 0:
                # base case for the forward part
                prev_f_sum = start_prob[st]
            else:
                prev_f_sum = sum(f_prev[k]*trans_prob[k][st] for k,s in enumerate(states))

            f_curr[st] = emm_prob[st][observation_i] * prev_f_sum

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd = sum(f_curr[k] * trans_prob[k][end_st] for k,s in enumerate(states))

    # backward part of the algorithm
    bkw = []
    b_prev = np.zeros((len(states)))
    for i, observation_i_plus in enumerate(reversed(np.append(observations[1:],0))):
        b_curr = np.zeros((len(states)))
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
    # print(len(bkw))
    # print(len(fwd))
    for i in range(len(observations)):
        for st,s in enumerate(states):
            posterior[i,st] = fwd[i][st] * bkw[i][st] / p_fwd
        # posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st,s in enumerate(states)})
    assert p_fwd-p_bkw < 1e-5
    return fwd, bkw, posterior
