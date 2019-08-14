import numpy as np

class Assimilator:
    def __init__(self, pomdp, windowLength):
        self.pomdp = pomdp
        self.windowLength = windowLength if windowLength is not None else len(self.measurements)
        self.viterbiPaths = {}
        self.viterbiProbabilities = {}

    def update_belief(self, belief, actionIdx, obs):
        T = self.pomdp.transition_matrix_for_action(actionIdx)
        E = self.pomdp.emission_matrix_for_action(obs,actionIdx)

        new_belief = np.multiply(E,(np.transpose(T).dot(belief)))
        new_belief = new_belief / np.linalg.norm(new_belief,1)
        return new_belief


    def fwd_bkwd(self, start_prob = None, firstMeasurement = 0, windowLength = None):
        def fwd(self,fv,ev):
            for s, state in enumerate(pomdp.states):
                fv = np.multiply(pomdp.emissionMat[:,ev],np.transpose(pomdp.transitionMat).dot(fv).flatten())
                c = np.linalg.norm(fv,1)
            return fv.flatten()/c, c

        def bkwd(self,b,ev):
            for s, state in enumerate(pomdp.states):
                b[s] = np.multiply(pomdp.emissionMat[:,ev],b).dot(pomdp.transitionMat[s,:])
            return b.flatten()
        windowLength = windowLength if windowLength is not None else self.windowLength
        start_prob = start_prob if start_prob is not None else pomdp.initialProb


        ev = list(range(firstMeasurement,firstMeasurement+windowLength))
        states = pomdp.states

        # forward part of the algorithm
        fwd = np.zeros((windowLength,len(states)))
        c = np.zeros(windowLength)
        for i, observation_i in enumerate(ev):
            if i == 0:
                fwd[i,:],c[i] = self.fwd(start_prob,ev[i])
            else:
                fwd[i,:],c[i] = self.fwd(fwd[i-1,:],ev[i])

        # backward part of the algorithm
        sv = np.zeros((windowLength,len(states)))
        b = np.ones(len(states))
        for i, observation_i in reversed(list(enumerate(ev))):
            sv[i,:] = np.multiply(fwd[i,:],c[i]*b)
            sv[i,:] = sv[i,:]/np.linalg.norm(sv[i,:],1)
            b = self.bkwd(b,ev[i])

        self.fullposterior = sv
        return sv

    def computeMostLikelyPaths(self, posteriorMatrix, initialProb = None):
        initialProb = initialProb if initialProb is not None else pomdp.initialProb

        sizeofPos = len(posteriorMatrix)

        # add one to include initial states
        transitions = np.zeros((sizeofPos+1,len(pomdp.states)))
        probabilities = np.zeros((sizeofPos+1,len(pomdp.states)))

        probabilities[0,:] = initialProb.flatten()

        for step in range(1,sizeofPos+1):
            for i,s in enumerate(pomdp.states):
                transitions[step,i] = np.argmax(np.multiply(pomdp.transitionMat[:,i],probabilities[step-1,:]))
                probabilities[step,i] = pomdp.emissionMat[i,step-1]*np.max(np.multiply(pomdp.transitionMat[:,i],probabilities[step-1,:]))
            probabilities[step,:] = probabilities[step,:] / np.linalg.norm(probabilities[step,:],1)

        return transitions[1:,], probabilities[1:,]

    def decodeTransitions(self, transitions):
        paths = np.zeros((len(transitions)+1,len(pomdp.states)))
        for step in reversed(list(range(0,len(transitions)+1))):
            for i,s in enumerate(pomdp.states):
                if step is len(transitions):
                    paths[step,i] = i
                else:
                    paths[step,i] = transitions[step,int(paths[step+1,i])]

        return paths

    def windowedSmoother(self):
        currentposterior = np.zeros((self.windowLength+1, len(pomdp.states)))
        currentposterior[0,:] = pomdp.initialProb.flatten()
        pomdp.windowedPosteriors = {}
        pomdp.windowedPosteriors[0] = currentposterior

        self.viterbiTransitions = {}
        self.viterbiTransitions[0] = np.zeros((self.windowLength, len(pomdp.states)))
        self.viterbiTransitions[0][0,:] = np.argmax(pomdp.initialProb)*np.ones((1,len(pomdp.states)))

        self.viterbiProbabilities = {}
        self.viterbiProbabilities[0] = np.zeros((self.windowLength+1, len(pomdp.states)))
        self.viterbiProbabilities[0][0,:] = pomdp.initialProb.flatten()

        self.viterbiPaths = {}
        self.viterbiPaths[0] = np.zeros((self.windowLength+1, len(pomdp.states)))
        self.viterbiPaths[0][0,:] = range(0,len(pomdp.states))

        #measurement index
        mIdx = 0
        while mIdx+self.windowLength-1 < len(pomdp.measurements):
            currentposterior = self.fwd_bkwd(currentposterior[0,:], mIdx) # windowLength x nStates
            self.windowedPosteriors[mIdx+1] = np.concatenate((self.windowedPosteriors[mIdx][0:mIdx+1,:],currentposterior))

            # Viterbi to compute most likely path
            # transitions, probabilities = self.computeMostLikelyPaths(currentposterior, self.viterbiProbabilities[mIdx][mIdx,:]) # windowLength+1 x nStates
            transitions, probabilities = self.computeMostLikelyPaths(self.windowedPosteriors[mIdx+1][1:,:]) # windowLength+1 x nStates

            # self.viterbiTransitions[mIdx+1] = np.vstack((self.viterbiTransitions[mIdx][0:mIdx,:],transitions))
            # self.viterbiProbabilities[mIdx+1] = np.vstack((self.viterbiProbabilities[mIdx][0:mIdx+1,:],probabilities))
            self.viterbiTransitions[mIdx+1] = transitions
            self.viterbiProbabilities[mIdx+1] = probabilities
            self.viterbiPaths[mIdx+1] = self.decodeTransitions(self.viterbiTransitions[mIdx+1])
            mIdx += 1

    def fullSmoother(self):
        self.fullPosterior = self.fwd_bkwd(pomdp.initialProb, 0, windowLength = len(pomdp.measurements))
        self.fullPosterior = np.concatenate((np.transpose(pomdp.initialProb),self.fullPosterior))

        # Viterbi to compute most likely path
        transitions, probabilities = self.computeMostLikelyPaths(self.fullPosterior[1:,:])

        self.optimalPath = self.decodeTransitions(transitions)
        self.optimalProbabilities = probabilities
