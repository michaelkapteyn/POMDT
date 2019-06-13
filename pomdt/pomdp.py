import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from pomdt.damageLibrary import *
from pomdt.measurementGenerator import *
from pomdt.utils import *


class POMDP:
    def __init__(self, nTimeSteps, states, actions, measurementGenerator, groundTruthState, initialBelief=None, transitionMat=None, emissionMat=None):
        self.step = 0
        self.nTimeSteps = nTimeSteps

        self.measurementGenerator = measurementGenerator
        self.noisymeasurements = measurementGenerator.nMeasurements
        self.states = states
        self.actions = actions
        self.groundTruthState = groundTruthState

        self.initialBelief = initialBelief if initialBelief is not None else self.defaultInitialBelief()
        self.transitionMat = transitionMat
        # self.emissionMat = emissionMat if emissionMat is not None else self.defaultEmissionMatrix()
        self.measurement_covariance = 50*50
        self.Reward = self.defaultReward

    def emission_matrix_for_action(self, obs, action):
        mIdx = self.step-1
        m = obs
        sigma = np.sqrt(self.measurement_covariance)
        E = np.zeros((len(self.states),1))

        for stateIdx, state in enumerate(self.states):
            ### this should be cleaned up
            stateIdxTuple = []
            for componentIdx, componentState in enumerate(state):
                stateIdxTuple.append(list(self.measurementGenerator.dlib.states.values())[componentIdx].index(componentState))
            stateIdxTuple = tuple(stateIdxTuple)
            ### this should be cleaned up

            predictedMeasurements = self.measurementGenerator.getMeasurement(stateIdxTuple,action, noisy=False)
            for sensIdx in range(self.measurementGenerator.nMeasurements):
                E[stateIdx,0] = E[stateIdx,0] + np.log(norm.pdf(m[sensIdx], predictedMeasurements[sensIdx], sigma))
            E[stateIdx,0] = np.exp(E[stateIdx,0])
        E[:,0] = E[:,0]/np.linalg.norm(E[:,0],1)
        return E

    def transition_matrix_for_action(self, action):
        if action == 0:
            T =  self.transitionMat["3g"]
        elif action == 1:
            T = self.transitionMat["2g"]
        return T

    def get_action(self, belief):
        if np.argmax(belief) < len(belief)/2.0:
            return 1
        else:
            return 0

    def take_action(self, si, actionIdx):
        self.step = self.step+1
        # perform an action
        # transition the underlying groundtruth state, return sj
        self.state = self.groundTruthState[self.step]

        # compute the next observation
        obs = self.measurementGenerator.getMeasurement(self.state, actionIdx)

        # return the reward obtained R(action,si,sj,action):
        reward = self.Reward(actionIdx, si, self.state, obs)
        return self.state, obs, reward

    def defaultReward(self,actionIdx, si, sj, obs):
        return 1

    def defaultInitialBelief(self):
        initialProb = np.zeros((len(self.states),1))
        initialProb[0] = 1
        return initialProb

    def defaultTransitionMatrix(self, alpha):
        tp = np.zeros((1,len(self.states)))
        tp[0,0]= 1-alpha
        tp[0,1] = alpha

        transition_probability = tp
        for i in range(1,len(self.states)):
           transition_probability = np.vstack((transition_probability,np.roll(tp,i)))

        transition_probability[-1,-1] = 1
        transition_probability[-1,0] = 0

        transition_probability = transition_probability + 0.00
        for i in range(0,len(self.states)):
            transition_probability[i,:] = transition_probability[i,:] / np.linalg.norm(transition_probability[i,:] ,1)
        print(transition_probability)
        assert(False)
        return transition_probability

    def defaultEmissionMatrix(self):
        sigma = np.sqrt(self.measurement_covariance)
        E = np.ones((len(self.states),len(self.measurements)))
        for stateIdx, state in enumerate(self.states):
            for mIdx, m in enumerate(self.measurements):
                for sensIdx in range(self.damageLibrary.nSensors):
                    E[stateIdx,mIdx] = E[stateIdx,mIdx]*norm.pdf(m[sensIdx], self.damageLibrary.predictedMeasurements[state][sensIdx], sigma)
        for mIdx, m in enumerate(self.measurements):
            E[:,mIdx] = E[:,mIdx]/np.linalg.norm(E[:,mIdx],1)
        return E
