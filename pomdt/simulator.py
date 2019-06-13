from pomdt.assimilator import *
import matplotlib.pyplot as plt

class Simulator:
    def __init__(self, pomdp, visualizer):
        self.visualizer = visualizer
        self.pomdp = pomdp
        self.assimilator = Assimilator(pomdp, 1)
        # self.params = Params()

    def run(self):
        # visualiser = self.visualizer
        pomdp = self.pomdp
        assimilator = self.assimilator
        total_rewards= 0

        belief = pomdp.initialBelief
        state = pomdp.groundTruthState[0]


        print('~~~ Initializing Simulation ~~~')
        print('''
        ++++++++++++++++++++++
            Starting State:  {}
            Init Belief: {}
            Time Horizon: {}
            Max Duration: {}
        ++++++++++++++++++++++'''.format(state, belief, pomdp.step, 199))

        for i in range(199):
            # plan, take action and receive environment feedbacks
            actionIdx = pomdp.get_action(belief)
            print("action is {}".format(actionIdx))
            state, obs, reward = pomdp.take_action(state, actionIdx)

            # update states
            belief = assimilator.update_belief(belief, actionIdx, obs)
            self.visualizer.update(belief,state)
            # plt.show()
            plt.pause(0.01)
            total_rewards += reward

            # print info
            print('\n'.join([
            'Taking action: {}'.format(self.pomdp.actions[actionIdx]),
            'Observation: {}'.format(obs),
            'Reward: {}'.format(reward),
            'New state: {}'.format(state),
            'New Belief:']))
            for state, prob in zip(self.pomdp.states, belief):
                print("(" + state[0] + "," + state[1]+ ") : " + str(prob))
            print('=' * 20)


        print('{} steps taken. Total reward = {}'.format(i + 1, total_rewards))
        return
