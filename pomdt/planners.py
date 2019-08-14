import numpy as np
import matplotlib.pyplot as plt

class Naive:
    def getPolicy(self):
        def naivepolicy(belief):
            if np.argmax(belief) < len(belief)/2.0:
                return 1
            else:
                return 0
        return naivepolicy

class QMDP:
    def __init__(self, pomdp):
        self.pomdp = pomdp

    def getPolicy(self):
        pomdp = self.pomdp
        r = pomdp.reward
        gamma = pomdp.gamma
        maxit = 100
        Ns = len(pomdp.states)
        Na = len(pomdp.actions)

        V = np.zeros((Ns,maxit))
        v = np.zeros((Na,1))
        for it in range(2,100):
            for sIdx, s in enumerate(pomdp.stateIdxs):
                for aIdx, a in enumerate(pomdp.actionIdxs):
                    v[aIdx] = r(s,a) + np.dot(np.transpose(V[:,it-1]),pomdp.transition_matrix_for_action(aIdx)[sIdx,:]);
                V[sIdx,it]  = gamma*np.max(v);
        V =  V[:,-1];

        Q  =  np.zeros((Ns,Na));
        for sIdx, s in enumerate(pomdp.stateIdxs):
            for aIdx, a in enumerate(pomdp.actionIdxs):
                Q[sIdx,aIdx] = r(s,a) + np.dot(V,pomdp.transition_matrix_for_action(aIdx)[sIdx,:]);

        bestQ = np.max(Q,1)
        self.bestU = np.argmax(Q,1);

        def qmdppolicy(belief):
            print(self.bestU)
            print(np.dot(belief.flatten(),self.bestU))
            return int(round(np.dot(belief.flatten(),self.bestU)))

        return qmdppolicy


class AlphaVector(object):
    """
    Simple wrapper for the alpha vector used for representing the value function for a POMDP as a piecewise-linear,
    convex function
    """
    def __init__(self, a, v):
        self.action = a
        self.v = v

    def copy(self):
        return AlphaVector(self.action, self.v)

class PBVI:
    def __init__(self, T, pomdp):
        self.model = pomdp
        self.alpha_vecs = [AlphaVector(a=-1, v=np.zeros(len(self.model.states)))] # filled with a dummy alpha vector
        self.solved = False

        self.belief_points = [np.array(b)/np.sum(b) for b in [[np.random.uniform() for s in self.model.states] for p in np.arange(0, 10)]]
        # self.belief_points = [np.array((1., 0., 0., 0.)),np.array((0., 1., 0., 0.)),np.array((0., 0., 1., 0.)),np.array((0., 0., 0., 1.))]
        self.compute_gamma_reward()
        self.solve(T)

    def getPolicy(self):
        return self.get_action

    def compute_gamma_reward(self):
        """
        :return: Action_a => Reward(s,a) matrix
        """
        self.gamma_reward = {}
        for a,i in enumerate(self.model.actions):
            self.gamma_reward[a] = np.array([self.model.reward_function(a, s) for s,i in enumerate(self.model.states)])

    def compute_gamma_action_obs(self, a, o):
        """
        Computes a set of vectors, one for each previous alpha
        vector that represents the update to that alpha vector
        given an action and observation
        :param a: action index
        :param o: observation index
        """
        m = self.model
        gamma_action_obs = []
        for alpha in self.alpha_vecs:
            v = np.zeros(len(m.states))  # initialize the update vector [0, ... 0]
            for i, si in enumerate(m.states):
                for j, sj in enumerate(m.states):
                    if i == o:
                        v[i] += m.transition_matrix_for_action(a)[i,j] * \
                            1.0 * \
                            alpha.v[j]
                            # m.emission_matrix_for_action(a)[i, o] * \
                    else:
                        v[i] += 0
                v[i] *= m.discount
            gamma_action_obs.append(v)
        return gamma_action_obs

    def solve(self, T):
        if self.solved:
            return

        m = self.model
        for step in range(T):
            print(step)
            # First compute a set of updated vectors for every action/observation pair
            # Action(a) => Observation(o) => UpdateOfAlphaVector (a, o)
            gamma_intermediate = {
                a: {
                    o: self.compute_gamma_action_obs(a, o)
                    for o,s in enumerate(m.states)
                } for a,s in enumerate(m.actions)
            }

            # Now compute the cross sum
            gamma_action_belief = {}
            for a,s in enumerate(m.actions):

                gamma_action_belief[a] = {}
                for bidx, b in enumerate(self.belief_points):

                    gamma_action_belief[a][bidx] = self.gamma_reward[a].copy()

                    for o,s in enumerate(m.states):
                        # only consider the best point
                        best_alpha_idx = np.argmax(np.dot(gamma_intermediate[a][o], b))
                        print(gamma_intermediate[a][o][best_alpha_idx])
                        gamma_action_belief[a][bidx] += gamma_intermediate[a][o][best_alpha_idx]

            # Finally compute the new(best) alpha vector set
            self.alpha_vecs, max_val = [], -np.inf

            for bidx, b in enumerate(self.belief_points):
                best_av, best_aa = None, None

                for a,i in enumerate(m.actions):
                    val = np.dot(gamma_action_belief[a][bidx], b)
                    if best_av is None or val > max_val:
                        max_val = val
                        best_av = gamma_action_belief[a][bidx].copy()
                        best_aa = a

                self.alpha_vecs.append(AlphaVector(a=best_aa, v=best_av))
                print(self.model.actions)
            for i,alpha in enumerate(self.alpha_vecs):
                print("=======")
                print(self.belief_points[i])
                print(alpha.v)
                print(alpha.action)
                print("=======")

        fig = plt.figure()
        b1 = np.linspace(0,1,100)
        for alpha in self.alpha_vecs:
            plt.plot(b1, alpha.v[0]*b1+alpha.v[1]*(1-b1))
        plt.show()
        exit()
        self.solved = True

    def get_action(self, belief):
        max_v = -np.inf
        best = None
        for av in self.alpha_vecs:
            v = np.dot(av.v, belief)
            if v > max_v:
                max_v = v
                best = av
