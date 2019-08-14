import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pomdt.damageLibrary import *
from pomdt.measurementGenerator import *
from pomdt.pomdp import *
from pomdt.utils import *

class Visualizer():
    def __init__(self, pomdp):
        self.pomdp = pomdp
        self.fig = plt.figure(figsize=(10, 8))

    def initializePlotWindow(self):
        pomdp = self.pomdp
        # fig = plt.figure(figsize=(10, 8))
        self.ax1 = plt.subplot(2,1,1)
        # self.plotMeasurements([0,5,10], self.ax1, pomdp.cleanmeasurements)
        self.ax2 = plt.subplot(2,1,2,xlim=(0, pomdp.nTimeSteps), ylim=(-1, len(pomdp.states)))
        # Change axis labels
        plt.yticks(range(0, len(pomdp.states)), [i for i in pomdp.states], rotation=0)
        self.fig.subplots_adjust(left=0.25)
        plt.ylabel('State in Damage Library')
        plt.xlabel('Time step')
        self.plotGroundTruthState(self.ax2)
        posteriorScatter = self.ax2.scatter(0,0, color="black",label="Posterior Densities")
        # plt.show()
        # plt.pause()

    def plotMeasurements(self, indices, ax=None, cleanmeasurements=None):
        model = self.pomdp
        ax = ax or plt.gca()
        measurementcolors = ['red','green','blue']
        for plotIdx, sensorIdx in enumerate(indices):
            if cleanmeasurements is not None:
                plt.plot(range(model.nTimeSteps), cleanmeasurements[:,sensorIdx],color='black')
            plt.scatter(range(model.nTimeSteps), model.measurements[:,sensorIdx],color=measurementcolors[plotIdx],label="sensor #"+str(sensorIdx))
        plt.legend()
        plt.ylabel('Measured Microstrain')
        # plt.show()

    def plotGroundTruthState(self, ax = None):
        model = self.pomdp
        ax = ax or plt.gca()
        plt.plot(range(0,model.nTimeSteps),model.groundTruthState,'--',color = "black", label= "Ground Truth")

    def plotFullSmoother(self, ax = None):
        model = self.pomdp
        ax = ax or plt.gca()
        mostlikelyidx = np.argmax(model.optimalProbabilities[-1,:])
        plt.plot(range(0,model.nTimeSteps+1),model.optimalPath[:,mostlikelyidx], color="red")

    def addbelieftoplot(self, belief, step):
        model = self.pomdp
        xpts = np.repeat(step,len(model.states))
        ypts = range(0,len(model.states))
        sizes = 4*belief.flatten()
        self.ax2.scatter(xpts,ypts,sizes, color="black",label="Posterior Densities")

    def animateWindowedSmoother(model):
        # Initialize figure
        fig = plt.figure(figsize=(10, 8))
        ax1 = plt.subplot(2,1,1)
        plotMeasurements(model,[0,5,10], ax1, model.cleanmeasurements)
        ax = plt.subplot(2,1,2,xlim=(0, model.nTimeSteps), ylim=(-1, len(model.states)))

        # Change axis labels
        plt.yticks(range(0, len(model.states)), [i for i in model.states], rotation=0)
        fig.subplots_adjust(left=0.25)
        plt.ylabel('State in Damage Library')
        plt.xlabel('Time step')

        # plot reference lines for each state
        for i,s in enumerate(model.states):
            plt.plot([0,model.nTimeSteps],[i,i],linewidth=0.1,color="grey")

        # plot the ground truth state
        plotGroundTruthState(model, ax)

        # create plot objects for animated lines (values will be initialized later)
        currentWindowLine, = ax.plot(0, 0 ,color="grey", label="Window Estimate")
        historyLine, = ax.plot(0, 0, color="black", label="Estimate History")
        posteriorScatter = plt.scatter(0,0, color="black",label="Posterior Densities")

        # Add Legend
        plt.legend(loc='lower right')


        def updatePosteriors(pos, model, timestep):
            xpts = np.repeat(range(0,timestep+model.windowLength-1),len(model.states))
            ypts = np.tile(range(0,len(model.states)), timestep+model.windowLength-1)
            sizes = 4*model.windowedPosteriors[timestep].flatten()
            pos.set_offsets(np.transpose(np.vstack((xpts,ypts))))
            pos.set_sizes(sizes)
            return pos,

        def updateWindowLine(line, model, timestep, statistic="viterbi"):
            if statistic is "viterbi":
                if timestep is 0:
                    timestep += 1
                mostlikelyidx = np.argmax(model.viterbiProbabilities[timestep][-1,:])
                line.set_data(np.array(range(timestep,timestep+model.windowLength)),np.array(model.viterbiPaths[timestep][timestep:,mostlikelyidx]))
            elif statistic is "mode":
                #plot mode of posterior at each timestep (not the joint posterior!)
                line.set_data(range(timestep,timestep+model.windowLength),np.argmax(model.windowedPosteriors[timestep][timestep-1:,:],1))
            elif statistic is "mean":
                #plot mean of posterior at each timestep (not the joint posterior!)
                statevec = range(0,len(model.states))
                line.set_data(range(timestep,timestep+model.windowLength),model.windowedPosteriors[timestep][timestep-1:,:].dot(statevec))
            else:
                pass
            return line,

        def updateHistoryLine(line, model, timestep, statistic="viterbi"):
            if statistic is "viterbi":
                mostlikelyidx = np.argmax(model.viterbiProbabilities[timestep][-1,:])
                line.set_data(range(0,timestep),model.viterbiPaths[timestep][0:timestep,mostlikelyidx])
            elif statistic is "mode":
                #plot mode of posterior at each timestep (not the joint posterior!)
                line.set_data(range(0,timestep),np.argmax(model.windowedPosteriors[timestep][0:timestep,:],1))
            elif statistic is "mean":
                #plot mean of posterior at each timestep (not the joint posterior!)
                statevec = range(0,len(model.states))
                line.set_data(range(0,timestep),model.windowedPosteriors[timestep][0:timestep,:].dot(statevec))
            else:
                pass
            return line,


        def init():  # only required for blitting to give a clean slate.
            # updateWindowLine(currentWindowLine ,model, 0, "viterbi")
            # updateHistoryLine(historyLine, model, 0, "viterbi")
            updatePosteriors(posteriorScatter,model,0)

            return currentWindowLine, historyLine, posteriorScatter,

        def animate(i):
            #update the current window line
            # updateWindowLine(currentWindowLine, model, i, "viterbi")
            #update the line plot outside of the window
            # updateHistoryLine(historyLine, model, i, "viterbi")

            #update the posterior bubbles
            updatePosteriors(posteriorScatter ,model, i)

            # if i is model.nTimeSteps-model.windowLength+1:
                # plotFullSmoother(model)

            return currentWindowLine, historyLine, posteriorScatter

        ani = animation.FuncAnimation(fig, animate, init_func=init, interval=100, repeat=False, blit=True, frames=model.nTimeSteps-model.windowLength+2)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('smoothing.mp4', writer=writer)
        plt.show()
