import matplotlib
matplotlib.use("macOSX")
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D #3d plots

from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.image import BboxImage,imread
from matplotlib.transforms import Bbox

from pomdt.damageLibrary import *
from pomdt.measurementGenerator import *
from pomdt.pomdp import *
from pomdt.utils import *

class Visualizer():
    def __init__(self, pomdp, states, posterior_fig=None, sensor_fig=None, policy_fig=None, action_fig=None, state_fig=None, capability_fig=None,):

        self.pomdp = pomdp
        self.states = states
        self.components = list(self.states.keys())
        self.xstates = self.states[self.components[0]]
        self.ystates = self.states[self.components[1]]
        self.nx = len(self.xstates)
        self.ny = len(self.ystates)
        self.idx = -1
        data = np.zeros((self.ny,self.nx))
        data[0,0] = 1

        ## Posterior
        self.posterior_fig = posterior_fig
        if self.posterior_fig is not None:
            self.gs = GridSpec(4,4)
            self.ax_joint = self.posterior_fig.add_subplot(self.gs[1:4,0:3])
            self.ax_marg_x = self.posterior_fig.add_subplot(self.gs[0,0:3])
            self.ax_marg_y = self.posterior_fig.add_subplot(self.gs[1:4,3])
            self.posterior_fig.subplots_adjust(left = 0.2, bottom = 0.3)

        ## Sensor
        self.sensor_fig = sensor_fig
        if self.sensor_fig is not None:
            self.ax_sensor = self.sensor_fig.add_subplot(1,1,1)
            self.sensor_fig.subplots_adjust(left = 0.2, bottom = 0.2)

        ## Policy
        self.policy_fig = policy_fig
        if self.policy_fig is not None:
            self.ax_policy = self.policy_fig.add_subplot(1,1,1)
            self.policy_fig.subplots_adjust(left = 0.2, bottom = 0.2)

        ## Action History
        self.action_fig = action_fig
        if self.action_fig is not None:
            self.ax_action = self.action_fig.add_subplot(1,1,1)
            self.action_fig.subplots_adjust(left = 0.15, bottom = 0.2)

        ## State History
        self.state_fig = state_fig
        if self.state_fig is not None:
            self.ax_state = []
            for c in range(0, len(pomdp.actions)):
                self.ax_state.append(self.state_fig.add_subplot(1,2,c+1))
                self.ax_state[c].set_xlabel('Time Step')
                self.ax_state[c].set_ylabel('States')
                self.ax_state[c].set_xlim([1, self.pomdp.nTimeSteps])
                self.ax_state[c].set_ylim([-0.1, 4.1])
            self.state_fig.subplots_adjust(left = 0.2, bottom = 0.2,wspace=0.6)

        self.capability_fig = capability_fig
        if self.capability_fig is not None:
            self.ax_capability = self.capability_fig.add_subplot(1,1,1)
            self.ax_capability.axis('off')


    def plot_all(self, belief, gtstate, obs, cleanobs):
        self.idx += 1
        data = np.transpose(np.reshape(belief,(self.ny,self.nx)))
        marg_x = np.sum(data,0)
        marg_y = np.sum(data,1)
        marg_x = marg_x/np.sum(marg_x)
        marg_y = marg_y/np.sum(marg_y)

        if self.posterior_fig is not None:
            self.plot_posterior(data, marg_x, marg_y, gtstate)

        if self.sensor_fig is not None:
            self.plot_sensor(obs, cleanobs)

        if self.policy_fig is not None:
            self.plot_policy()

        if self.action_fig is not None:
            self.plot_actions()

        if self.state_fig is not None:
            self.plot_states(marg_x, marg_y)

        if self.capability_fig is not None:
            self.plot_capability(data)



    def plot_posterior(self, data, marg_x, marg_y, gtstate):
        ###### Joint
        self.ax_joint.cla()
        self.gt = self.ax_joint.scatter(gtstate[0],gtstate[1], s=100, color="red")
        self.joint = self.ax_joint.imshow(data, interpolation='nearest',
                            origin='bottom',
                            aspect='auto', # get rid of this to have equal aspect
                            vmin=np.min(0),
                            vmax=np.max(1),
                            cmap='Blues')

        self.ax_joint.set_xticks(range(0, len(self.xstates)))
        self.ax_joint.set_yticks(range(0, len(self.ystates)))

        xlabels=[]
        ylabels=[]
        for s in self.xstates:
            xlabels.append(s[5:])
        for s in self.ystates:
            ylabels.append(s[5:])
        self.ax_joint.set_xticklabels(xlabels, rotation = 90)
        self.ax_joint.set_yticklabels(ylabels, rotation = 0)

        self.ax_joint.set_xlabel('sec02 state')
        self.ax_joint.set_ylabel('sec04 state')


        ###### Marginals
        self.ax_marg_x.cla()
        self.ax_marg_y.cla()


        self.marg_x = self.ax_marg_x.bar(np.arange(self.nx),marg_x, color=cm.Blues(marg_x), width=1)
        self.marg_y = self.ax_marg_y.barh(np.arange(self.ny),marg_y, color=cm.Blues(marg_y), height=1)
        self.ax_marg_x.plot([gtstate[0],gtstate[0]],[0,1],color="red")
        self.ax_marg_y.plot([0,1],[gtstate[1],gtstate[1]],color="red")

        plt.setp(self.ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(self.ax_marg_y.get_yticklabels(), visible=False)
        plt.setp(self.ax_marg_y.get_xticklabels(), visible=False)
        plt.setp(self.ax_marg_x.get_yticklabels(), visible=False)

        self.ax_marg_x.set_xlim([-0.5, 4.5])
        self.ax_marg_x.set_ylim([0, 1])

        self.ax_marg_y.set_xlim([0, 1])
        self.ax_marg_y.set_ylim([-0.5, 4.5])


    def plot_sensor(self, obs, cleanobs):
        indices = [0,10,20]
        measurementcolors = ['red','green','blue']
        for plotIdx, sensorIdx in enumerate(indices):
            self.ax_sensor.scatter(self.idx, cleanobs[sensorIdx], color='black')
            self.ax_sensor.scatter(self.idx, obs[sensorIdx], color=measurementcolors[plotIdx],label="sensor #"+str(sensorIdx))

        self.ax_sensor.set_xlabel('Time Step')
        self.ax_sensor.set_ylabel('Measured Microstrain')
        self.ax_sensor.set_xlim([1, self.pomdp.nTimeSteps])

    def plot_policy(self):
        self.ax_policy.set_xlabel('sec02')
        self.ax_policy.set_ylabel('sec04')
        self.ax_policy.imshow(np.reshape(self.pomdp.solver.bestU,(5,5)),origin='lower',cmap='Blues')


    def plot_actions(self):
        self.ax_action.scatter(self.idx, self.pomdp.actionhistory[self.idx-1], color='black')
        self.ax_action.set_ylim([-0.1, 1.1])
        self.ax_action.set_xlabel('Time Step')
        self.ax_action.set_ylabel('Action taken')
        self.ax_action.set_yticks(np.arange(0.,1.1,1.))
        self.ax_action.set_xlim([1, self.pomdp.nTimeSteps])
        self.ax_action.set_ylim([-0.1, 1.1])
        labels = [item.get_text() for item in self.ax_action.get_yticklabels()]
        labels[0] = '2g'
        labels[1] = '3g'
        self.ax_action.set_yticklabels(labels)


    def plot_states(self, marg_x, marg_y):
        c0 = np.dot(np.arange(len(marg_x)),marg_x)
        c1 = np.dot(np.arange(len(marg_y)),marg_y)
        self.ax_state[0].scatter(self.idx, c0, color='black', s=0.5)
        self.ax_state[1].scatter(self.idx, c1, color='black', s=0.5)

        self.ax_state[0].scatter(self.idx, self.pomdp.statehistory[self.idx-1][0], color='blue',s = 0.5)
        self.ax_state[1].scatter(self.idx, self.pomdp.statehistory[self.idx-1][1], color='blue',s = 0.5)

        self.ax_state[0].set_ylim([-0.1, 4.1])
        self.ax_state[1].set_ylim([-0.1, 4.1])


    def plot_capability(self, data):
        self.ax_capability.cla()
        estimated_capability = np.sum(np.multiply(data.flatten()/np.sum(data),self.pomdp.capability.flatten()))-0.0109556963769677
        # self.estimated_capability = self.ax_capability_dist.plot([estimated_capability,estimated_capability],[0,0.1], color="red")
        self.ax_capability.axis('off')
        self.estimated_capability_text = self.ax_capability.text(0.05,0.4,"Max. Load Factor: "+ str(round(estimated_capability,2)), color="black",fontsize=20,transform=self.ax_capability.transAxes)

    # def add_image_labels(self,ax):
    #     bbox_imagex = []
    #     bbox_imagey = []
    #     for i,TICKXPOS in enumerate([0, 1., 2., 3.,4.]):
    #         TICKYPOS = -1.0
    #         lowerCorner = ax.transData.transform((TICKXPOS-.2,TICKYPOS-.2))
    #         upperCorner = ax.transData.transform((TICKXPOS+.2,TICKYPOS+.2))
    #
    #         bbox_imagex.append(BboxImage(Bbox([[lowerCorner[0],
    #                                      lowerCorner[1]],
    #                                      [upperCorner[0],
    #                                      upperCorner[1]],
    #                                      ]),
    #                                norm = None,
    #                                origin=None,
    #                                clip_on=False,
    #                                ))
    #         bbox_imagex[i].set_data(imread('images/sec02_'+ str(i)+'.png'))
    #         ax.add_artist(bbox_imagex[i])
    #
    #     for i,TICKYPOS in enumerate([0, 1., 2., 3.,4.]):
    #         TICKXPOS = -1.3
    #         lowerCorner = ax.transData.transform((TICKXPOS-.2,TICKYPOS-.2))
    #         upperCorner = ax.transData.transform((TICKXPOS+.2,TICKYPOS+.2))
    #
    #         bbox_imagey.append(BboxImage(Bbox([[lowerCorner[0],
    #                                      lowerCorner[1]],
    #                                      [upperCorner[0],
    #                                      upperCorner[1]],
    #                                      ]),
    #                                norm = None,
    #                                origin=None,
    #                                clip_on=False,
    #                                ))
    #
    #         bbox_imagey[i].set_data(imread('images/sec04_' + str(i)+'.png'))
    #         ax.add_artist(bbox_imagey[i])
