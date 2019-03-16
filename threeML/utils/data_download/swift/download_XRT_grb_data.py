import urllib3
import sys, os
import time
import requests


def download_Swift_XRT_grb_data(obs_id, tstart, tstop, mode='BOTH', filename='xrt.tar.gz', destination_dir='.'):
    """FIXME! briefly describe function

    :param obs_id: 
    :param tstart: 
    :param tstop: 
    :param mode: 
    :param filename: 
    :param destination_dir: 
    :returns: 
    :rtype: 

    """
    

    # set the time slice to analyze
    timeslices = '%f-%f' % (tstart, tstop)

    # build the form
    data = dict(
        targ=id,
        name=id,
        pubpriv=1,
        public=1,
        z=0,
        sno=0,
        name1='a',
        time1=timeslices,
        mode1=mode,
        name2='',
        time2='',
        mode2=mode,
        name3='',
        time3='',
        mode3=mode,
        name4='',
        time4='',
        mode4=mode,
        grade0='0',
    )

    print('requesting build...')

    r = requests.post("http://www.swift.ac.uk/xrt_spectra/build_slice_spec.php", data=data)

    url = r.url

    r2 = r
    itr = 0
    while 'This page will be reloaded in 30 seconds' in r2.content:
        print 'sleeping 30 seconds ...'
        time.sleep(30)
        if itr == 0:
            print('requesting %s ...' % url)
        r2 = requests.get(url)

        itr += 1

    print('downloading: %s' % os.path.join(url, filename))
    connection_pool = urllib3.PoolManager()
    resp = connection_pool.request('GET', os.path.join(url, 'a.tar.gz'))
    ''

    f = open(os.path.join(destination_dir, filename), 'wb')
    f.write(resp.data)
    f.close()
    resp.release_conn()

    tar = tarfile.open(os.path.join(destination_dir, filename))

    tar.extractall(path=destination_dir)

    modes = dict(
        pc=dict(
            source=os.path.join(destination_dir, 'apcsource.pi'),
            bak=os.path.join(destination_dir, 'apcbak.pi'),
            rmf=os.path.join(destination_dir, 'apc.rmf'),
            arf=os.path.join(destination_dir, 'apc.arf')),
        wt=dict(
            source=os.path.join(destination_dir, 'awtsource.pi'),
            bak=os.path.join(destination_dir, 'awtbak.pi'),
            rmf=os.path.join(destination_dir, 'awt.rmf'),
            arf=os.path.join(destination_dir, 'awt.arf')))

    return modes
