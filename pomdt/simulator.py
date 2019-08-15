from pomdt.assimilator import *
import matplotlib.pyplot as plt

class Simulator:
    def __init__(self, pomdp, assimilator, visualizer):
        self.pomdp = pomdp
        self.assimilator = assimilator
        self.visualizer = visualizer
        self.initialize_sim()

    def run(self):
        for i in range(self.pomdp.nTimeSteps):
            self.advance()
            plt.pause(0.01)

        print('{} steps taken. Total reward = {}'.format(i + 1, total_rewards))

    def initialize_sim(self):
        self.total_rewards= 0

        self.belief = self.pomdp.initialBelief
        self.state = self.pomdp.groundTruthState[0]


        print('~~~ Initializing Simulation ~~~')
        print('''
        ++++++++++++++++++++++
            Starting State:  {}
            Init Belief: {}
            Time Horizon: {}
            Max Duration: {}
        ++++++++++++++++++++++'''.format(self.state, self.belief, self.pomdp.step, self.pomdp.nTimeSteps))

    def advance(self):
        # plan, take action and receive environment feedbacks
        actionIdx = self.pomdp.get_action(self.belief)
        print("action is {}".format(actionIdx))
        self.state, obs, reward = self.pomdp.take_action(self.state, actionIdx)

        # update states
        self.belief = self.assimilator.update_belief(self.belief, actionIdx, obs)

        cleanobs = self.pomdp.measurementGenerator.getMeasurement(self.state, actionIdx, noisy=False)
        self.visualizer.plot_all(self.belief, self.state, obs, cleanobs)

        self.total_rewards += reward

        # print info
        print('\n'.join([
        'Taking action: {}'.format(self.pomdp.actions[actionIdx]),
        'Observation: {}'.format(obs),
        'Reward: {}'.format(reward),
        'New state: {}'.format(self.state),
        'New Belief:']))
        for state, prob in zip(self.pomdp.states, self.belief):
            print("(" + state[0] + "," + state[1]+ ") : " + str(prob))
        print('=' * 20)
