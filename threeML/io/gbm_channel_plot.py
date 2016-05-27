from step_plot import step_plot
import matplotlib.pyplot as plt
import numpy as np
def gbm_channel_plot(chan_min,chan_max,counts,**keywords):

    chans = np.array(zip(chan_min, chan_max))
    width = chan_max - chan_min
    fig = plt.figure(666)
    ax = fig.add_subplot(111)
    step_plot(chans, counts / width, ax, **keywords)
    ax.set_xscale('log')
    ax.set_yscale('log')

    return ax


def excluded_channel_plot(chan_min,chan_max,mask,counts,bkg,ax):


    # Figure out the best limit
    chans = np.array(zip(chan_min, chan_max))
    width = chan_max - chan_min

    top = max([max(bkg/width) , max(counts/width) ])
    top = top+top*.5
    bottom = min([min(bkg/width) , min(counts/width) ])
    bottom = bottom-bottom*.2
    
    # Find the contiguous regions
    slices = slice_disjoint((~mask).nonzero()[0])

    for region in slices:

        ax.fill_between([chan_min[region[0]],chan_max[region[1]]],
                        bottom,
                        top,
                        color='k',
                        alpha=.5)

    ax.set_ylim(bottom,top)

    




def slice_disjoint(arr):
    slices=[]
    startSlice = 0
    
    for i in range(len(arr)-1): 
        if arr[i+1]>arr[i]+1:
            endSlice = arr[i]
            slices.append([startSlice,endSlice])
            startSlice=arr[i+1]
    if endSlice!=arr[-1]:
        slices.append([startSlice,arr[-1]])
    return slices
