def step_plot(xbins,y,ax,color='b',lw=1.,ls='-',fill=False,fillAlpha=1.,**keywords):
    '''
    Routine for plotting a in steps with the ability to fill the plot
    xbins is a 2D list of start and stop values.
    y are the values in the bins.
    '''

    x=[]
    newy=[]
    for t,v in zip(xbins,y):
        x.append(t[0])
        newy.append(v)
        x.append(t[1])
        newy.append(v)
    if fill:
        ax.fill_between(x,newy,0,color=color,linewidth=lw,linestyle=ls,alpha=fillAlpha)
    else:
        ax.plot(x,newy,color=color,linewidth=lw,linestyle=ls,**keywords)
    
