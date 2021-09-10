```python
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp

np.seterr(all="ignore")

from threeML import *
from threeML.io.package_data import get_path_of_data_file
from threeML.io import update_logging_level
from threeML.utils.data_download.Fermi_LAT.download_LAT_data import LAT_dataset
from astropy.io import fits as pyfits

SMALL_SIZE = 15
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#This if you want to toggle different type of logging level.
update_logging_level("INFO")
log.error('error')
log.info('info')
log.debug('debug')

```

    Welcome to JupyROOT 6.22/06


    
    WARNING RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
    
    
    WARNING RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
    
    
    WARNING RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
    
    Warning: cannot import _healpy_pixel_lib module


    [[31mERROR   [0m][31m error[0m
    [[32mINFO    [0m][32m info[0m


## GtBurst
Gtburst contains all the classes and methods to perform Fermi LAT data. It internally uses the official fermitools software. Here an example to list the IRFS available:


```python
from GtBurst import IRFS
irfs = IRFS.IRFS.keys()
print(irfs)
```

    odict_keys(['p7rep_transient', 'p7rep_source', 'p7rep_clean', 'p7rep_ultraclean', 'p8r2_transient100e', 'p8r2_transient100', 'p8r2_transient020e', 'p8r2_transient020', 'p8r2_transient010e', 'p8r2_transient010', 'p8r2_source', 'p8r2_clean', 'p8r2_ultraclean', 'p8r2_ultracleanveto', 'p8r2_transient100s', 'p8r2_transient015s', 'p8_transient100e', 'p8_transient100', 'p8_transient020e', 'p8_transient020', 'p8_transient010e', 'p8_transient010', 'p8_source', 'p8_clean', 'p8_ultraclean', 'p8_ultracleanveto', 'p8_sourceveto', 'p8_transient100s', 'p8_transient015s'])


## The LAT Transient Builder
Let's see how to make a plug in for the unbinned analysis of Fermi LAT data. First we use the information form a triggered GRB to obtain MET, RA and DEC, that are needed for the analysis.


```python
from GtBurst.TriggerSelector import TriggerSelector
myFavoriteGRB = 'bn190114873'
def findGRB(grb_name):
    a=TriggerSelector()
    a.downloadList()
    myGRB={}
    for x in a.data: 
        if x[0]==myFavoriteGRB:
            myGRB['MET']=float(x[1])
            myGRB['RA']=float(x[3])
            myGRB['DEC']=float(x[4])
            myGRB['ERR']=float(x[5])
            return myGRB
            pass
    return None
```


```python
myGRB=findGRB(myFavoriteGRB)
print(myGRB)
```

    {'MET': 569192227.626, 'RA': 54.51, 'DEC': -26.939, 'ERR': 0.05}


Then, we download LAT data and we build the transient builder, we want to analyze 1000 seconds since the trigger. Let's start download the data:


```python
tstart                = 0
tstop                 = 1000
```


```python
myLATdataset = LAT_dataset()

myLATdataset.make_LAT_dataset(
    ra                    = myGRB['RA'],
    dec                   = myGRB['DEC'],
    radius                = 12,
    trigger_time          = myGRB['MET'],
    tstart                = tstart,
    tstop                 = tstop,
    data_type             = "Extended",
    destination_directory = '../FermiData',
    Emin= 100.,    
    Emax= 10000.0
) # Energies are MeV (this is from 100 MeV to 10 GeV)
```

We want perform a time resolved analysis. So, first we look at the data. We can play with the ROI selection and the cut.


```python
roi       = 10
zmax      = 110.
thetamax  = 180.0
irfs      = 'p8_transient020e'
strategy  = 'time'
myLATdataset.extract_events(roi, zmax, irfs, thetamax, strategy='time')
```

    time -p gtmktime scfile=/Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit sctable="SC_DATA" filter="(DATA_QUAL>0 || DATA_QUAL==-1) && LAT_CONFIG==1 && IN_SAA!=T && LIVETIME>0 && (ANGSEP(RA_ZENITH,DEC_ZENITH,54.51,-26.939)<=(110.0-10))" roicut=no evfile=/Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/FermiData/bn190114873/gll_ft1_tr_bn190114873_v00.fit evtable="EVENTS" outfile="gll_ft1_tr_bn190114873_v00_mkt.fit" apply_filter=yes overwrite=no header_obstimes=yes tstart=569192227.626 tstop=569193227.626 gtifile="default" chatter=2 clobber=yes debug=no gui=no mode="ql"
    real 0.09
    user 0.06
    sys 0.02
    
    Using 305 data
    
    time -p gtselect infile=gll_ft1_tr_bn190114873_v00_mkt.fit outfile=gll_ft1_tr_bn190114873_v00_filt.fit ra=54.51 dec=-26.939 rad=10.0 tmin=569192227.626 tmax=569193227.626 emin=100.0 emax=10000.0 zmin=0.0 zmax=110.0 evclass=8 evtype=3 convtype=-1 phasemin=0.0 phasemax=1.0 evtable="EVENTS" chatter=2 clobber=yes debug=no gui=no mode="ql"
    Done.
    real 0.10
    user 0.07
    sys 0.02
    
    Selected 251 events.
    [[32mINFO    [0m][32m Extracted 251 events[0m


Once we are happy, we can bin the light curve. We will perform a likelihood analysi in each bin, so we need to make sure there are enough events in each bin. In the following example we select 10 event per bin. We could also select less event, but you should always consider the number of free parameters in your model. The number of events in each bin should always be larger than the number of free parameters.


```python
%matplotlib inline
event_file = pyfits.open(myLATdataset.filt_file)
event_times = sorted(event_file['EVENTS'].data['TIME']-myGRB['MET'])
intervals=event_times[0::10]
plt.hist(event_times);
plt.hist(event_times,intervals,histtype='step')
plt.show()
```


    
![png](output_12_0.png)
    


tstarts and tstops are defined as strings, with somma separated values for the starts and the ends of the time bins: For example tsrats="0,1,10" and tstops="1,10,20". To convert arrays in string we use these few lines of code:


```python
tstarts=tstops=''
for t0,t1 in zip(intervals[:-1],intervals[1:]):
    tstarts+='%.4f,' % t0
    tstops +='%.4f,' % t1
    pass
tstarts = tstarts[:-1].replace('-','\\-')
tstops  = tstops[:-1].replace('-','\\-')
```

We can now make an instance the LAT transient builder


```python
analysis_builder = TransientLATDataBuilder(myLATdataset.grb_name,
                                           outfile=myLATdataset.grb_name,
                                           roi=roi,
                                           tstarts=tstarts,
                                           tstops=tstops,
                                           irf=irfs,
                                           zmax=zmax,
                                           galactic_model='template',
                                           particle_model='isotr template',
                                           datarepository='../FermiData')
analysis_builder.display()
```

    outfile                                                       190114873
    roi                                                                  10
    tstarts               2.6996,3.6358,3.9968,4.4024,4.7375,5.0909,5.54...
    tstops                3.6358,3.9968,4.4024,4.7375,5.0909,5.5471,5.98...
    zmax                                                                 11
    emin                                                                  1
    emax                                                                  1
    irf                                                    p8_transient020e
    galactic_model                                                 template
    particle_model                                           isotr template
    tsmin                                                                 2
    strategy                                                           time
    thetamax                                                             18
    spectralfiles                                                        no
    liketype                                                       unbinned
    optimizeposition                                                     no
    datarepository                                             ../FermiData
    ltcube                                                                 
    expomap                                                                
    ulphindex                                                            -2
    flemin                                                              100
    flemax                                                            10000
    fgl_mode                                                           fast
    filter_GTI                                                        False
    likelihood_profile                                                False
    remove_fits_files                                                 False
    dtype: object



```python
tstops
```




    '3.6358,3.9968,4.4024,4.7375,5.0909,5.5471,5.9896,6.3998,6.6889,7.0117,7.2936,7.7731,8.2167,8.8763,9.6573,10.5680,12.0568,14.6165,17.7834,21.4962,30.0798,40.8747,48.7118,73.7262,172.5754'



The run method will run (using gtburst) all the fermitools needed to obtain the needed file for the likelihood analysis (livetimecubes, exposure maps. It will also perfom a simple likelihood analysis with the standard likelihood of the fermitools (pylikelihood). The dataproducts created here will be used by threeML to make the fit.


```python
LAT_observations = analysis_builder.run(include_previous_intervals = True)
```

    About to run the following command:
    /Users/omodei/miniconda/envs/threeml_ixpe_fermi/lib/python3.7/site-packages/fermitools/GtBurst/scripts/doTimeResolvedLike.py 190114873 --outfile '190114873' --roi 10.000000 --tstarts '2.6996,3.6358,3.9968,4.4024,4.7375,5.0909,5.5471,5.9896,6.3998,6.6889,7.0117,7.2936,7.7731,8.2167,8.8763,9.6573,10.5680,12.0568,14.6165,17.7834,21.4962,30.0798,40.8747,48.7118,73.7262' --tstops '3.6358,3.9968,4.4024,4.7375,5.0909,5.5471,5.9896,6.3998,6.6889,7.0117,7.2936,7.7731,8.2167,8.8763,9.6573,10.5680,12.0568,14.6165,17.7834,21.4962,30.0798,40.8747,48.7118,73.7262,172.5754' --zmax 110.000000 --emin 100.000000 --emax 100000.000000 --irf 'p8_transient020e' --galactic_model 'template' --particle_model 'isotr template' --tsmin 20.000000 --strategy 'time' --thetamax 180.000000 --spectralfiles 'no' --liketype 'unbinned' --optimizeposition 'no' --datarepository '../FermiData' --ltcube '' --expomap '' --ulphindex -2.000000 --flemin 100.000000 --flemax 10000.000000 --fgl_mode 'fast'   
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval10.568-12.0568/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval12.0568-14.6165/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval14.6165-17.7834/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval17.7834-21.4962/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval2.6996-3.6358/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval21.4962-30.0798/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval3.6358-3.9968/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval3.9968-4.4024/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval30.0798-40.8747/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval4.4024-4.7375/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval4.7375-5.0909/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval40.8747-48.7118/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval48.7118-73.7262/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval5.0909-5.5471/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval5.5471-5.9896/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval5.9896-6.3998/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval6.3998-6.6889/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval6.6889-7.0117/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval7.0117-7.2936/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval7.2936-7.7731/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval7.7731-8.2167/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval73.7262-172.5754/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval8.2167-8.8763/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval8.8763-9.6573/gll_ft2_tr_bn190114873_v00.fit
    The ft2 file does not exist. Please examine!
    we will grab the data file for you.
    copied ../FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit to interval9.6573-10.568/gll_ft2_tr_bn190114873_v00.fit
    
    Requested intervals:
    ------------------------------------------------------
    2.6996               - 3.6358
    3.6358               - 3.9968
    3.9968               - 4.4024
    4.4024               - 4.7375
    4.7375               - 5.0909
    5.0909               - 5.5471
    5.5471               - 5.9896
    5.9896               - 6.3998
    6.3998               - 6.6889
    6.6889               - 7.0117
    7.0117               - 7.2936
    7.2936               - 7.7731
    7.7731               - 8.2167
    8.2167               - 8.8763
    8.8763               - 9.6573
    9.6573               - 10.568
    10.568               - 12.0568
    12.0568              - 14.6165
    14.6165              - 17.7834
    17.7834              - 21.4962
    21.4962              - 30.0798
    30.0798              - 40.8747
    40.8747              - 48.7118
    48.7118              - 73.7262
    73.7262              - 172.5754
    
    Data files:
    -----------
    eventfile            /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/FermiData/bn190114873/gll_ft1_tr_bn190114873_v00.fit
    ft2file              /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/FermiData/bn190114873/gll_ft2_tr_bn190114873_v00.fit
    rspfile              /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/FermiData/bn190114873/gll_cspec_tr_bn190114873_v00.rsp
    cspecfile            /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/FermiData/bn190114873/gll_cspec_tr_bn190114873_v00.pha
    
    ROI:
    -----
    R.A.                 54.51
    Dec.                 -26.939
    Radius               10.0
    
    Interval # 1 (2.6996-3.6358):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval2.6996-3.6358 already exists, skipping
    
    Interval # 2 (3.6358-3.9968):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval3.6358-3.9968 already exists, skipping
    
    Interval # 3 (3.9968-4.4024):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval3.9968-4.4024 already exists, skipping
    
    Interval # 4 (4.4024-4.7375):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval4.4024-4.7375 already exists, skipping
    
    Interval # 5 (4.7375-5.0909):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval4.7375-5.0909 already exists, skipping
    
    Interval # 6 (5.0909-5.5471):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval5.0909-5.5471 already exists, skipping
    
    Interval # 7 (5.5471-5.9896):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval5.5471-5.9896 already exists, skipping
    
    Interval # 8 (5.9896-6.3998):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval5.9896-6.3998 already exists, skipping
    
    Interval # 9 (6.3998-6.6889):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval6.3998-6.6889 already exists, skipping
    
    Interval # 10 (6.6889-7.0117):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval6.6889-7.0117 already exists, skipping
    
    Interval # 11 (7.0117-7.2936):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval7.0117-7.2936 already exists, skipping
    
    Interval # 12 (7.2936-7.7731):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval7.2936-7.7731 already exists, skipping
    
    Interval # 13 (7.7731-8.2167):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval7.7731-8.2167 already exists, skipping
    
    Interval # 14 (8.2167-8.8763):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval8.2167-8.8763 already exists, skipping
    
    Interval # 15 (8.8763-9.6573):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval8.8763-9.6573 already exists, skipping
    
    Interval # 16 (9.6573-10.568):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval9.6573-10.568 already exists, skipping
    
    Interval # 17 (10.568-12.0568):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval10.568-12.0568 already exists, skipping
    
    Interval # 18 (12.0568-14.6165):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval12.0568-14.6165 already exists, skipping
    
    Interval # 19 (14.6165-17.7834):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval14.6165-17.7834 already exists, skipping
    
    Interval # 20 (17.7834-21.4962):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval17.7834-21.4962 already exists, skipping
    
    Interval # 21 (21.4962-30.0798):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval21.4962-30.0798 already exists, skipping
    
    Interval # 22 (30.0798-40.8747):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval30.0798-40.8747 already exists, skipping
    
    Interval # 23 (40.8747-48.7118):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval40.8747-48.7118 already exists, skipping
    
    Interval # 24 (48.7118-73.7262):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval48.7118-73.7262 already exists, skipping
    
    Interval # 25 (73.7262-172.5754):
    -----------------------
    
    /Users/omodei/GRBWorkDir/MY_PYTHON_MODULES/gitrepository/threeML/sandbox/LATTransientBuilderExample/interval73.7262-172.5754 already exists, skipping


At this point we can create the FermiLATLike plugins from each of the observation: 


```python
LAT_plugins={}     
for l in LAT_observations:
    LAT_name = 'LAT_%06.3f-%06.3f' % (float(l.tstart),float(l.tstop))
    LAT_plugins[LAT_name] = l.to_LATLike()
    pass
```

For reference, these are the keys save in the dictionary. 


```python
LAT_plugins.keys()
```




    dict_keys(['LAT_10.568-12.057', 'LAT_12.057-14.617', 'LAT_14.617-17.783', 'LAT_17.783-21.496', 'LAT_02.700-03.636', 'LAT_21.496-30.080', 'LAT_03.636-03.997', 'LAT_03.997-04.402', 'LAT_30.080-40.875', 'LAT_04.402-04.737', 'LAT_04.737-05.091', 'LAT_40.875-48.712', 'LAT_48.712-73.726', 'LAT_05.091-05.547', 'LAT_05.547-05.990', 'LAT_05.990-06.400', 'LAT_06.400-06.689', 'LAT_06.689-07.012', 'LAT_07.012-07.294', 'LAT_07.294-07.773', 'LAT_07.773-08.217', 'LAT_73.726-172.575', 'LAT_08.217-08.876', 'LAT_08.876-09.657', 'LAT_09.657-10.568'])



Now we can perform the fit in each bin. Note that we set the model, and we set some initial values. All the resulting joint likelihood objects are stored in a dictioonary to be used later for plotting.


```python
results = {}
update_logging_level("DEBUG")

for T0,T1 in zip(intervals[:-1],intervals[1:]):
    GRB = PointSource('GRB',ra=myGRB['RA'],dec=myGRB['DEC'],spectral_shape=Powerlaw_flux())
    model = Model(GRB)
    model.GRB.spectrum.main.Powerlaw_flux.a = 100.0*u.MeV
    model.GRB.spectrum.main.Powerlaw_flux.b = 10000.0*u.MeV
    model.GRB.spectrum.main.Powerlaw_flux.F = 1.0
    LAT_name       = 'LAT_%06.3f-%06.3f' % (T0,T1)
    LAT_model_name = ('LAT%dX%d' % (T0,T1)).replace('-','n')
    datalist = DataList(LAT_plugins[LAT_name])
    model['GRB.spectrum.main.Powerlaw_flux.F'].bounds = (1e-6,1e6)
    model['GRB.spectrum.main.Powerlaw_flux.F'].value = 1e-2
    model['GRB.spectrum.main.Powerlaw_flux.index'].value = -2.2
    model['GRB.spectrum.main.Powerlaw_flux.index'].bounds = (-3,0)
    jl=JointLikelihood(model,datalist,verbose=False)
    model[LAT_model_name+'_GalacticTemplate_Value'].value=1.0
    model[LAT_model_name+'_GalacticTemplate_Value'].fix=True
    model[LAT_model_name+'_GalacticTemplate_Value'].fix=True
    #model.display( complete=True )
    jl.set_minimizer('ROOT')
    jl.fit(compute_covariance=True)
    results[LAT_name] = jl
    pass
```

    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 93.92 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(9.2 -2.5 +3.4) x 10^-3</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-3.0000 +/- 0.0031</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT2X3_IsotropicTemplate_Normalization</th>
      <td>1.5 +/- 3.1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table9706761232">
<tr><td>1.00</td><td>-0.00</td><td>-0.01</td></tr>
<tr><td>-0.00</td><td>1.00</td><td>0.00</td></tr>
<tr><td>-0.01</td><td>0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT2X3</th>
      <td>40.6745</td>
    </tr>
    <tr>
      <th>total</th>
      <td>40.6745</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>79.349</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>81.349</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 74.2 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(2.4 -0.6 +0.9) x 10^-2</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-3.000 +/- 0.015</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT3X3_IsotropicTemplate_Normalization</th>
      <td>1.50 +/- 0.05</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10831191760">
<tr><td>1.00</td><td>-0.01</td><td>-0.00</td></tr>
<tr><td>-0.01</td><td>1.00</td><td>-0.00</td></tr>
<tr><td>-0.00</td><td>-0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT3X3</th>
      <td>29.5252</td>
    </tr>
    <tr>
      <th>total</th>
      <td>29.5252</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>57.0505</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>59.0505</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[31mERROR   [0m][31m cannot setLAT3X4_IsotropicTemplate_Normalization to 1.5651077244237945[0m
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 71.66 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(2.1 -0.6 +0.8) x 10^-2</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-2.98 +/- 0.32</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT3X4_IsotropicTemplate_Normalization</th>
      <td>1.1 +/- 0.7</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table7784110800">
<tr><td>1.00</td><td>-0.19</td><td>-0.00</td></tr>
<tr><td>-0.19</td><td>1.00</td><td>0.00</td></tr>
<tr><td>-0.00</td><td>0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT3X4</th>
      <td>19.8912</td>
    </tr>
    <tr>
      <th>total</th>
      <td>19.8912</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>37.7824</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>39.7824</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 54.52 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(2.3 -0.7 +0.9) x 10^-2</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-2.5 +/- 0.4</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT4X4_IsotropicTemplate_Normalization</th>
      <td>(5.0 +/- 0.5) x 10^-1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10831431568">
<tr><td>1.00</td><td>-0.31</td><td>-0.00</td></tr>
<tr><td>-0.31</td><td>1.00</td><td>0.00</td></tr>
<tr><td>-0.00</td><td>0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT4X4</th>
      <td>21.0058</td>
    </tr>
    <tr>
      <th>total</th>
      <td>21.0058</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>40.0116</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>42.0116</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 51.28 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(2.0 -0.6 +0.8) x 10^-2</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-2.2 +/- 0.4</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT4X5_IsotropicTemplate_Normalization</th>
      <td>(5.0 +/- 0.6) x 10^-1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10831813584">
<tr><td>1.00</td><td>-0.31</td><td>0.00</td></tr>
<tr><td>-0.31</td><td>1.00</td><td>-0.00</td></tr>
<tr><td>0.00</td><td>-0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT4X5</th>
      <td>28.5349</td>
    </tr>
    <tr>
      <th>total</th>
      <td>28.5349</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>55.0698</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>57.0698</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 49.24 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(1.09 -0.32 +0.4) x 10^-2</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-1.36 +/- 0.28</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT5X5_IsotropicTemplate_Normalization</th>
      <td>(5.00 +/- 0.14) x 10^-1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10831814288">
<tr><td>1.00</td><td>-0.25</td><td>0.00</td></tr>
<tr><td>-0.25</td><td>1.00</td><td>-0.00</td></tr>
<tr><td>0.00</td><td>-0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT5X5</th>
      <td>24.1777</td>
    </tr>
    <tr>
      <th>total</th>
      <td>24.1777</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>46.3554</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>48.3554</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 51.32 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(1.6 -0.4 +0.6) x 10^-2</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-2.1 +/- 0.4</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT5X5_IsotropicTemplate_Normalization</th>
      <td>(5.0 +/- 0.5) x 10^-1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10831735632">
<tr><td>1.00</td><td>-0.31</td><td>0.00</td></tr>
<tr><td>-0.31</td><td>1.00</td><td>-0.00</td></tr>
<tr><td>0.00</td><td>-0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT5X5</th>
      <td>22.7373</td>
    </tr>
    <tr>
      <th>total</th>
      <td>22.7373</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>43.4746</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>45.4746</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 54.17999999999999 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(1.9 -0.5 +0.7) x 10^-2</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-2.5 +/- 0.4</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT5X6_IsotropicTemplate_Normalization</th>
      <td>1.2 +/- 0.7</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table6754635920">
<tr><td>1.00</td><td>-0.30</td><td>-0.00</td></tr>
<tr><td>-0.30</td><td>1.00</td><td>0.00</td></tr>
<tr><td>-0.00</td><td>0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT5X6</th>
      <td>25.8991</td>
    </tr>
    <tr>
      <th>total</th>
      <td>25.8991</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>49.7982</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>51.7982</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 49.46 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(2.5 -0.7 +0.9) x 10^-2</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-1.94 +/- 0.31</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT6X6_IsotropicTemplate_Normalization</th>
      <td>(5.000 +/- 0.012) x 10^-1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table7784133584">
<tr><td>1.00</td><td>-0.30</td><td>0.00</td></tr>
<tr><td>-0.30</td><td>1.00</td><td>-0.00</td></tr>
<tr><td>0.00</td><td>-0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT6X6</th>
      <td>24.347</td>
    </tr>
    <tr>
      <th>total</th>
      <td>24.347</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>46.6939</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>48.6939</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 54.04 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(2.0 -0.6 +0.9) x 10^-2</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-2.3 +/- 0.4</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT6X7_IsotropicTemplate_Normalization</th>
      <td>1.5 +/- 0.5</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10832117584">
<tr><td>1.00</td><td>-0.31</td><td>0.00</td></tr>
<tr><td>-0.31</td><td>1.00</td><td>-0.00</td></tr>
<tr><td>0.00</td><td>-0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT6X7</th>
      <td>21.1192</td>
    </tr>
    <tr>
      <th>total</th>
      <td>21.1192</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>40.2384</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>42.2384</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 51.68000000000001 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(2.3 -0.7 +0.9) x 10^-2</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-1.88 +/- 0.32</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT7X7_IsotropicTemplate_Normalization</th>
      <td>(5.00 +/- 0.13) x 10^-1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10832563792">
<tr><td>1.00</td><td>-0.30</td><td>-0.00</td></tr>
<tr><td>-0.30</td><td>1.00</td><td>0.00</td></tr>
<tr><td>-0.00</td><td>0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT7X7</th>
      <td>27.6851</td>
    </tr>
    <tr>
      <th>total</th>
      <td>27.6851</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>53.3702</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>55.3702</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 49.82 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(1.3 -0.4 +0.5) x 10^-2</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-1.82 +/- 0.31</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT7X7_IsotropicTemplate_Normalization</th>
      <td>(5.00 +/- 0.16) x 10^-1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10832691536">
<tr><td>1.00</td><td>-0.30</td><td>0.00</td></tr>
<tr><td>-0.30</td><td>1.00</td><td>-0.00</td></tr>
<tr><td>0.00</td><td>-0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT7X7</th>
      <td>33.4063</td>
    </tr>
    <tr>
      <th>total</th>
      <td>33.4063</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>64.8126</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>66.8126</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 62.6 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(1.8 -0.5 +0.7) x 10^-2</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-2.6 +/- 0.5</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT7X8_IsotropicTemplate_Normalization</th>
      <td>1.5 +/- 0.5</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10832473296">
<tr><td>1.00</td><td>-0.31</td><td>-0.00</td></tr>
<tr><td>-0.31</td><td>1.00</td><td>0.00</td></tr>
<tr><td>-0.00</td><td>0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT7X8</th>
      <td>30.0043</td>
    </tr>
    <tr>
      <th>total</th>
      <td>30.0043</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>58.0085</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>60.0085</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 97.89999999999999 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(1.08 -0.31 +0.4) x 10^-2</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-2.2 +/- 0.4</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT8X8_IsotropicTemplate_Normalization</th>
      <td>(0.1 +/- 1.7) x 10</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10832611536">
<tr><td>1.00</td><td>-0.31</td><td>-0.01</td></tr>
<tr><td>-0.31</td><td>1.00</td><td>0.01</td></tr>
<tr><td>-0.01</td><td>0.01</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT8X8</th>
      <td>28.292</td>
    </tr>
    <tr>
      <th>total</th>
      <td>28.292</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>54.5839</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>56.5839</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 50.519999999999996 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(1.00 -0.27 +0.4) x 10^-2</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-2.18 +/- 0.35</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT8X9_IsotropicTemplate_Normalization</th>
      <td>1.5000 +/- 0.0008</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10833044880">
<tr><td>1.00</td><td>-0.31</td><td>-0.00</td></tr>
<tr><td>-0.31</td><td>1.00</td><td>0.00</td></tr>
<tr><td>-0.00</td><td>0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT8X9</th>
      <td>42.0937</td>
    </tr>
    <tr>
      <th>total</th>
      <td>42.0937</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>82.1875</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>84.1875</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 49.419999999999995 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(6.8 -1.9 +2.7) x 10^-3</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-1.73 +/- 0.30</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT9X10_IsotropicTemplate_Normalization</th>
      <td>1.5000 +/- 0.0007</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10833148304">
<tr><td>1.00</td><td>-0.29</td><td>-0.00</td></tr>
<tr><td>-0.29</td><td>1.00</td><td>0.00</td></tr>
<tr><td>-0.00</td><td>0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT9X10</th>
      <td>43.3574</td>
    </tr>
    <tr>
      <th>total</th>
      <td>43.3574</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>84.7149</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>86.7149</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 49.0 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(3.8 -1.1 +1.5) x 10^-3</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-1.47 +/- 0.28</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT10X12_IsotropicTemplate_Normalization</th>
      <td>1.5000 +/- 0.0017</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10833267792">
<tr><td>1.00</td><td>-0.27</td><td>-0.00</td></tr>
<tr><td>-0.27</td><td>1.00</td><td>0.00</td></tr>
<tr><td>-0.00</td><td>0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT10X12</th>
      <td>44.7004</td>
    </tr>
    <tr>
      <th>total</th>
      <td>44.7004</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>87.4007</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>89.4007</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 55.42 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(2.7 -0.8 +1.1) x 10^-3</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-2.5 +/- 0.4</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT12X14_IsotropicTemplate_Normalization</th>
      <td>(5.00 +/- 0.17) x 10^-1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10833369232">
<tr><td>1.00</td><td>-0.31</td><td>0.00</td></tr>
<tr><td>-0.31</td><td>1.00</td><td>-0.00</td></tr>
<tr><td>0.00</td><td>-0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT12X14</th>
      <td>43.3579</td>
    </tr>
    <tr>
      <th>total</th>
      <td>43.3579</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>84.7158</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>86.7158</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 46.18 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(2.0 -0.6 +0.8) x 10^-3</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-1.78 +/- 0.30</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT14X17_IsotropicTemplate_Normalization</th>
      <td>1.1 +/- 0.7</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10833339856">
<tr><td>1.00</td><td>-0.29</td><td>-0.00</td></tr>
<tr><td>-0.29</td><td>1.00</td><td>0.00</td></tr>
<tr><td>-0.00</td><td>0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT14X17</th>
      <td>44.4271</td>
    </tr>
    <tr>
      <th>total</th>
      <td>44.4271</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>86.8543</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>88.8543</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 62.56 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(1.8 -0.5 +0.7) x 10^-3</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-1.73 +/- 0.29</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT17X21_IsotropicTemplate_Normalization</th>
      <td>(5 +/- 9) x 10^-1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table9706833040">
<tr><td>1.00</td><td>-0.29</td><td>-0.00</td></tr>
<tr><td>-0.29</td><td>1.00</td><td>0.00</td></tr>
<tr><td>-0.00</td><td>0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT17X21</th>
      <td>56.0886</td>
    </tr>
    <tr>
      <th>total</th>
      <td>56.0886</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>110.177</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>112.177</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 49.120000000000005 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(6.6 -1.8 +2.5) x 10^-4</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-1.42 +/- 0.27</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT21X30_IsotropicTemplate_Normalization</th>
      <td>(5.0 +/- 1.3) x 10^-1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10831200848">
<tr><td>1.00</td><td>-0.26</td><td>-0.00</td></tr>
<tr><td>-0.26</td><td>1.00</td><td>0.00</td></tr>
<tr><td>-0.00</td><td>0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT21X30</th>
      <td>56.9681</td>
    </tr>
    <tr>
      <th>total</th>
      <td>56.9681</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>111.936</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>113.936</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 49.32 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(5.0 -1.4 +1.9) x 10^-4</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-1.30 +/- 0.27</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT30X40_IsotropicTemplate_Normalization</th>
      <td>(5.0 +/- 0.5) x 10^-1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10834287696">
<tr><td>1.00</td><td>-0.23</td><td>-0.00</td></tr>
<tr><td>-0.23</td><td>1.00</td><td>0.00</td></tr>
<tr><td>-0.00</td><td>0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT30X40</th>
      <td>55.7441</td>
    </tr>
    <tr>
      <th>total</th>
      <td>55.7441</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>109.488</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>111.488</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 49.82 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(6.2 -1.8 +2.6) x 10^-4</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-1.33 +/- 0.28</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT40X48_IsotropicTemplate_Normalization</th>
      <td>(5.0 +/- 0.8) x 10^-1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10834341840">
<tr><td>1.00</td><td>-0.24</td><td>-0.00</td></tr>
<tr><td>-0.24</td><td>1.00</td><td>0.00</td></tr>
<tr><td>-0.00</td><td>0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT40X48</th>
      <td>47.3988</td>
    </tr>
    <tr>
      <th>total</th>
      <td>47.3988</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>92.7977</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>94.7977</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 49.84 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(3.3 -0.9 +1.2) x 10^-4</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-1.84 +/- 0.30</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT48X73_IsotropicTemplate_Normalization</th>
      <td>(5.0 +/- 1.2) x 10^-1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10834298448">
<tr><td>1.00</td><td>-0.32</td><td>-0.02</td></tr>
<tr><td>-0.32</td><td>1.00</td><td>0.01</td></tr>
<tr><td>-0.02</td><td>0.01</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT48X73</th>
      <td>69.0782</td>
    </tr>
    <tr>
      <th>total</th>
      <td>69.0782</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>136.156</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>138.156</td>
    </tr>
  </tbody>
</table>
</div>


    [[35mWARNING [0m][35m We have set the min_value of GRB.spectrum.main.Powerlaw_flux.F to 1e-99 because there was a postive transform[0m
    [[34mDEBUG   [0m][34m creating new MLE analysis[0m
    [[34mDEBUG   [0m][34m REGISTERING MODEL[0m
    
    Found Isotropic template for irf P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/iso_P8R3_TRANSIENT020E_V3_v1.txt
    
    Found Galactic template for IRF. P8R3_TRANSIENT020E_V3: /Users/omodei/miniconda/envs/threeml_ixpe_fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits
    
    Cutting the template around the ROI: 
    
    [[34mDEBUG   [0m][34m MODEL REGISTERED![0m
    [[32mINFO    [0m][32m set the minimizer to minuit[0m
    [[32mINFO    [0m][32m set the minimizer to ROOT[0m
    [[34mDEBUG   [0m][34m beginning the fit![0m
    [[34mDEBUG   [0m][34m starting local optimization[0m
    [[35mWARNING [0m][35m get_number_of_data_points not implemented, values for statistical measurements such as AIC or BIC are unreliable[0m
    [[35mWARNING [0m][35m 49.5 percent of samples have been thrown away because they failed the constraints on the parameters. This results might not be suitable for error propagation. Enlarge the boundaries until you loose less than 1 percent of the samples.[0m
    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(8.9 -2.8 +4) x 10^-5</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-1.87 +/- 0.35</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT73X172_IsotropicTemplate_Normalization</th>
      <td>1.5000 +/- 0.0007</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10834322128">
<tr><td>1.00</td><td>-0.36</td><td>0.02</td></tr>
<tr><td>-0.36</td><td>1.00</td><td>-0.01</td></tr>
<tr><td>0.02</td><td>-0.01</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT73X172</th>
      <td>79.7989</td>
    </tr>
    <tr>
      <th>total</th>
      <td>79.7989</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>157.598</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>159.598</td>
    </tr>
  </tbody>
</table>
</div>


You can usethis function to graphically display the results of your fit (folded model, data and residuals)


```python
i=3
T0,T1=intervals[i],intervals[i+1]
LAT_name       = 'LAT_%06.3f-%06.3f' % (T0,T1)
jl=results[LAT_name]
jl.results.display()
display_spectrum_model_counts(jl, step=False,figsize=(10,10));
```

    Best fit values:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>result</th>
      <th>unit</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.F</th>
      <td>(2.3 -0.7 +0.9) x 10^-2</td>
      <td>1 / (cm2 s)</td>
    </tr>
    <tr>
      <th>GRB.spectrum.main.Powerlaw_flux.index</th>
      <td>-2.5 +/- 0.4</td>
      <td></td>
    </tr>
    <tr>
      <th>LAT4X4_IsotropicTemplate_Normalization</th>
      <td>(5.0 +/- 0.5) x 10^-1</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    
    Correlation matrix:
    



<table id="table10831621328">
<tr><td>1.00</td><td>-0.31</td><td>-0.00</td></tr>
<tr><td>-0.31</td><td>1.00</td><td>0.00</td></tr>
<tr><td>-0.00</td><td>0.00</td><td>1.00</td></tr>
</table>


    
    Values of -log(likelihood) at the minimum:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>-log(likelihood)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAT4X4</th>
      <td>21.0058</td>
    </tr>
    <tr>
      <th>total</th>
      <td>21.0058</td>
    </tr>
  </tbody>
</table>
</div>


    
    Values of statistical measures:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statistical measures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AIC</th>
      <td>40.0116</td>
    </tr>
    <tr>
      <th>BIC</th>
      <td>42.0116</td>
    </tr>
  </tbody>
</table>
</div>


    
    WARNING MatplotlibDeprecationWarning: The 'nonposy' parameter of __init__() has been renamed 'nonpositive' since Matplotlib 3.3; support for the old name will be dropped two minor releases later.
    



    
![png](output_27_9.png)
    


We can see the evolution of the spectrum with time (not all the bins are diplayed):


```python
fig = plot_spectra(
    *[results[k].results for k in list(results.keys())[::2]],
    ene_min=100*u.MeV,ene_max=10*u.GeV,
    flux_unit="erg2/(cm2 s MeV)",
    energy_unit='MeV',
    fit_cmap="viridis",
    contour_cmap="viridis",
    contour_style_kwargs=dict(alpha=0.1)
);
fig.set_size_inches(10,10)
```


    processing MLE analyses:   0%|          | 0/13 [00:00<?, ?it/s]



    Propagating errors:   0%|          | 0/100 [00:00<?, ?it/s]



    Propagating errors:   0%|          | 0/100 [00:00<?, ?it/s]



    Propagating errors:   0%|          | 0/100 [00:00<?, ?it/s]



    Propagating errors:   0%|          | 0/100 [00:00<?, ?it/s]



    Propagating errors:   0%|          | 0/100 [00:00<?, ?it/s]



    Propagating errors:   0%|          | 0/100 [00:00<?, ?it/s]



    Propagating errors:   0%|          | 0/100 [00:00<?, ?it/s]



    Propagating errors:   0%|          | 0/100 [00:00<?, ?it/s]



    Propagating errors:   0%|          | 0/100 [00:00<?, ?it/s]



    Propagating errors:   0%|          | 0/100 [00:00<?, ?it/s]



    Propagating errors:   0%|          | 0/100 [00:00<?, ?it/s]



    Propagating errors:   0%|          | 0/100 [00:00<?, ?it/s]



    Propagating errors:   0%|          | 0/100 [00:00<?, ?it/s]


    [[34mDEBUG   [0m][34m converting MeV to MeV[0m



    
![png](output_29_15.png)
    


Finally, we can display flux lightcurves and index evolution with time.


```python
variates=['F','index']
y={}
for n in variates: 
    y[n]=[]
    y[n+'_p']=[]
    y[n+'_n']=[]
x=[]
dx=[]


for T0,T1 in zip(intervals[:-1],intervals[1:]):
    LAT_name       = 'LAT_%06.3f-%06.3f' % (T0,T1)
    x.append((T1+T0)/2)
    dx.append((T1-T0)/2)
    jl=results[LAT_name]
    res = jl.results
    mod = res.optimized_model
    ps  = mod.point_sources
    
    for n in variates:
        my_variate     = res.get_variates('GRB.spectrum.main.Powerlaw_flux.%s' % n)
        y[n].append(my_variate.median)
        y[n+'_p'].append(my_variate.equal_tail_interval()[1] - my_variate.median)
        y[n+'_n'].append(my_variate.median                   - my_variate.equal_tail_interval()[0])
        pass
    pass

fig=plt.figure(figsize=(10,15))
colors=['r','b']
ylabels=['Flux [100MeV - 10GeV] \n $\gamma$ cm$^{-2}$ s$^{-1}$','index']
for i,n in enumerate(variates):
    plt.subplot(len(variates)+1,1,i+1)
    plt.errorbar(x,y[n],xerr=dx,yerr=(y[n+'_n'],y[n+'_p']),ls='',c=colors[i])
    if i==0: plt.yscale('log')
    #plt.xscale('log')
    plt.ylabel(ylabels[i])
    pass
```


    
![png](output_31_0.png)
    



```python

```
