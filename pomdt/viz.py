import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib import cm

from pomdt.damageLibrary import *
from pomdt.measurementGenerator import *
from pomdt.pomdp import *
from pomdt.utils import *

class Visualizer():
    def __init__(self, pomdp, states):
        self.pomdp = pomdp
        self.states = states
        self.components = list(self.states.keys())
        self.xstates = self.states[self.components[0]]
        self.ystates = self.states[self.components[1]]
        self.nx = len(self.xstates)
        self.ny = len(self.ystates)

    def initializePlotWindow(self):
        self.fig = plt.figure(figsize=(10, 8))

        self.gs = GridSpec(4,4)

        data = np.zeros((self.ny,self.nx))
        data[0,0] = 1
        ##################
        ###### Joint
        ##################
        self.ax_joint = self.fig.add_subplot(self.gs[1:4,0:3])
        self.joint = self.ax_joint.imshow(data, interpolation='nearest',
                            origin='bottom',
                            aspect='auto', # get rid of this to have equal aspect
                            vmin=np.min(0),
                            vmax=np.max(1),
                            cmap='Blues')
        # cb = plt.colorbar(self.pos)

        self.gt = self.ax_joint.scatter(0,0, s=100, color="red")

        # Change axis labels
        self.ax_joint.set_xticks(range(0, len(self.xstates)))
        self.ax_joint.set_yticks(range(0, len(self.ystates)))
        self.ax_joint.set_xticklabels(self.xstates)
        self.ax_joint.set_yticklabels(self.ystates, rotation=0)

        self.ax_joint.set_xlabel('sec02 state')
        self.ax_joint.set_ylabel('sec04 state')

        ##################
        ###### Marginals
        ##################
        self.ax_marg_x = self.fig.add_subplot(self.gs[0,0:3])
        self.ax_marg_y = self.fig.add_subplot(self.gs[1:4,3])
        self.marg_x = self.ax_marg_x.bar(np.arange(self.nx),np.sum(data,0), width=1)
        self.marg_y = self.ax_marg_y.barh(np.arange(self.ny),np.sum(data,1), height=1)

        # Turn off tick labels on marginals
        plt.setp(self.ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(self.ax_marg_y.get_yticklabels(), visible=False)

        # set axis limits
        self.ax_marg_x.set_xlim([0, 4])
        self.ax_marg_x.set_ylim([0, 1])

        self.ax_marg_y.set_xlim([0, 4])
        self.ax_marg_y.set_ylim([0, 1])

    def update(self, belief, gtstate):
        data = np.transpose(np.reshape(belief,(self.ny,self.nx)))
        self.gt.set_offsets(gtstate)
        self.joint.set_data(data)
        self.ax_marg_x.cla()
        self.ax_marg_y.cla()
        marg_x = np.sum(data,0)
        marg_y = np.sum(data,1)

        self.marg_x = self.ax_marg_x.bar(np.arange(self.nx),marg_x/sum(marg_x), color=cm.Blues(marg_x/sum(marg_x)), width=1)
        self.marg_y = self.ax_marg_y.barh(np.arange(self.ny),marg_y/sum(marg_y), color=cm.Blues(marg_y/sum(marg_y)), height=1)
        self.ax_marg_x.plot([gtstate[0],gtstate[0]],[0,1],color="red")
        self.ax_marg_y.plot([0,1],[gtstate[1],gtstate[1]],color="red")

        plt.setp(self.ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(self.ax_marg_y.get_yticklabels(), visible=False)

        self.ax_marg_x.set_xlim([-0.5, 4.5])
        self.ax_marg_x.set_ylim([0, 1])

        self.ax_marg_y.set_xlim([0, 1])
        self.ax_marg_y.set_ylim([-0.5, 4.5])
