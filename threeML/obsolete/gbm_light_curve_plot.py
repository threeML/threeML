import matplotlib.pyplot as plt
import numpy as np
from threeML.io.step_plot import step_plot

def gbm_light_curve_plot(time_bins,cnts,bkg,width,selection):

    fig = plt.figure(777)
    ax  = fig.add_subplot(111)

    maxCnts = max(cnts/width)
    top =maxCnts+maxCnts*.2 
    minCnts = min(cnts[cnts>0]/width)
    bottom  = minCnts - minCnts*.2
    mean_time = map(np.mean,time_bins)
    
    step_plot(time_bins,cnts/width,ax,color='#8da0cb')

    ax.plot(mean_time,bkg,'#66c2a5',lw=2.)

    ax.fill_between(selection,bottom,top,color="#fc8d62",alpha=.4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (cnts/s)")
    ax.set_ylim(bottom,top)


