import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from pomdt import *


class POMDP:
    def __init__(self, nTimeSteps, states, stateIdxs, actions, actionIdxs, capability, measurementGenerator, groundTruthState, initialBelief=None, globaltransitionMat=None, localtransitionMat=None, emissionMat=None):
        self.step = 0
        self.nTimeSteps = nTimeSteps
        self.statetransitionmode = 'random'
        self.measurementGenerator = measurementGenerator
        self.noisymeasurements = measurementGenerator.nMeasurements
        self.states = states
        self.stateIdxs = stateIdxs
        self.actions = actions
        self.actionIdxs = actionIdxs
        self.actionhistory = []
        self.statehistory=[]
        self.capability = capability
        self.groundTruthState = groundTruthState

        self.initialBelief = initialBelief if initialBelief is not None else self.defaultInitialBelief()
        self.globaltransitionMat = globaltransitionMat
        self.localtransitionMat = localtransitionMat
        # self.emissionMat = emissionMat if emissionMat is not None else self.defaultEmissionMatrix()
        self.measurement_covariance = 40*40
        self.gamma = 0.9

        # solver = PBVI(self.nTimeSteps, self)
        # solver = Naive()
        self.solver = QMDP(self)
        self.policy = self.solver.getPolicy()


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
        T =  self.globaltransitionMat[action]
        return T

    def local_transition_matrix_for_action(self, component, action):
        T =  self.localtransitionMat[component][action]
        return T

    def get_action(self, belief):
        action = self.policy(belief)
        return action

    def take_action(self, si, actionIdx):
        self.step = self.step+1
        self.actionhistory.append(actionIdx)

        # perform an action
        # transition the underlying groundtruth state, return sj
        if self.statetransitionmode == 'Deterministic':
            self.state = self.groundTruthState[self.step]
        else:
            self.state = []
            for cIdx, st in enumerate(si):
                T = self.local_transition_matrix_for_action(cIdx, actionIdx)
                st = int(st)
                probabilities = T[st]
                self.state.append(np.random.choice(np.arange(0,len(probabilities)), 1, p=probabilities)[0])
            self.state = tuple(self.state)

        self.statehistory.append(self.state)
        # compute the next observation
        obs = self.measurementGenerator.getMeasurement(self.state, actionIdx)

        # return the reward obtained R(action,si,sj,action):
        reward = self.reward(si, actionIdx)
        return self.state, obs, reward

    def reward(self, stateIdx, actionIdx):
        s1 = float(stateIdx[0])
        s2 = float(stateIdx[1])
        a =  float(actionIdx)
        if s1==s2==4:
            return -5
        else:
            return 0.05*a + (np.power(8,3) -  np.power(s1+s2,3))/1000.

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
