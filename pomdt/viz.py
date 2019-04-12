import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from shmm.damageLibrary import *
from shmm.measurementGenerator import *
from shmm.shmm import *
from shmm.utils import *

def plotMeasurements(model, indices, ax=None, cleanmeasurements=None):
    ax = ax or plt.gca()
    measurementcolors = ['red','green','blue']
    for plotIdx, sensorIdx in enumerate(indices):
        if cleanmeasurements is not None:
            plt.plot(range(len(model.measurements)), cleanmeasurements[:,sensorIdx],color='black')
        plt.scatter(range(len(model.measurements)), model.measurements[:,sensorIdx],color=measurementcolors[plotIdx],label="sensor #"+str(sensorIdx))
    plt.legend()
    plt.ylabel('Measured Microstrain')
    # plt.show()

def plotGroundTruthState(model, ax = None, **kwargs):
    ax = ax or plt.gca()
    plt.plot(range(0,len(model.measurements)),model.groundTruthState,'--',**kwargs)

def animateWindowedSmoother(model):
    # Initialize figure
    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(2,1,1)
    plotMeasurements(model,[0,5,10], ax1, model.cleanmeasurements)
    ax = plt.subplot(2,1,2,xlim=(0, len(model.measurements)), ylim=(-1, len(model.states)))

    # Change axis labels
    plt.yticks(range(0, len(model.states)), [i.strip('defect__') for i in model.states], rotation=0)
    fig.subplots_adjust(left=0.25)
    plt.ylabel('State in Damage Library')
    plt.xlabel('Time step')

    # plot reference lines for each state
    for i,s in enumerate(model.states):
        plt.plot([0,len(model.measurements)],[i,i],linewidth=0.1,color="grey")

    # plot the ground truth state
    plotGroundTruthState(model, ax, color="black",)

    # create plot objects for animated lines (values will be initialized later)
    currentWindowLine, = ax.plot(0, 0 ,color="grey", label="Window Estimate")
    historyLine, = ax.plot(0, 0, color="black", label="Estimate History")
    posteriorScatter = plt.scatter(0,0, color="black",label="Posterior Densities")

    # Add Legend
    plt.legend()


    def updatePosteriors(pos, model, timestep):
        xpts = np.repeat(range(0,timestep+model.windowLength-1),len(model.states))
        ypts = np.tile(range(0,len(model.states)), timestep+model.windowLength-1)
        sizes = 4*model.windowedPosteriors[timestep].flatten()
        pos.set_offsets(np.transpose(np.vstack((xpts,ypts))))
        pos.set_sizes(sizes)
        return pos,

    def updateWindowLine(line, model, timestep, statistic="viterbi"):
        if statistic is "viterbi":
            mostlikelyidx = np.argmax(model.viterbiprobabilities[timestep][:,-1])
            line.set_data(np.array(range(timestep,timestep+model.windowLength)),np.array(model.viterbipaths[timestep][mostlikelyidx,timestep:]))
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
            mostlikelyidx = np.argmax(model.viterbiprobabilities[timestep][:,-1])
            line.set_data(range(0,timestep),model.viterbipaths[timestep][mostlikelyidx,0:timestep])
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
        updateWindowLine(currentWindowLine ,model, 0, "viterbi")
        updateHistoryLine(historyLine, model, 0, "viterbi")
        updatePosteriors(posteriorScatter,model,0)

        return currentWindowLine, historyLine, posteriorScatter,

    def animate(i):
        #update the current window line
        updateWindowLine(currentWindowLine, model, i, "viterbi")
        #update the line plot outside of the window
        updateHistoryLine(historyLine, model, i, "viterbi")

        #update the posterior bubbles
        updatePosteriors(posteriorScatter ,model, i)

        return currentWindowLine, historyLine, posteriorScatter,

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=100, repeat=False, blit=True, frames=len(model.measurements)-model.windowLength)
    plt.show()
