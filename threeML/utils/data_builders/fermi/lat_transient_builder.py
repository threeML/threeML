import collections
import os
import re
import shutil
import site
import subprocess
import uuid
from glob import glob

import pandas as pd
import yaml

from threeML.config import threeML_config
from threeML.io.file_utils import file_existing_and_readable
from threeML.io.logging import setup_logger

pd.reset_option('display.float_format')

log = setup_logger(__name__)


try: 
    from GtBurst import IRFS
    from GtBurst.Configuration import Configuration
    from GtBurst.FuncFactory import Spectra

    from threeML import FermiLATLike

    irfs = IRFS.IRFS.keys()
    spectra = Spectra()

    #irfs.append('auto')

    configuration = Configuration()

    has_fermitools = True

except (ImportError):

    has_fermitools = False

    if threeML_config.logging.startup_warnings:

        log.warning('No fermitools installed')
        




class LATLikelihoodParameter(object):

    def __init__(self, name, help_string, default_value=None, allowed_values=None, is_number=True, is_bool=False):
        """

        A container for the parameters that are needed by GtBurst

        :param name: the parameter name 
        :param help_string: the help string
        :param default_value: a default value if needed
        :param allowed_values: the values allowed for input
        :param is_number: if this is a number
        :param is_bool: if this is a bool
        :returns: 
        :rtype: 

        """

        self._name = name
        self._allowed_values = allowed_values
        self._default_value = default_value
        self._is_number = is_number
        self._is_bool = is_bool
        self._help_string = help_string
        self._is_set = False

        # if there is a default value, lets go ahead and set it

        if default_value is not None:
            self.__set_value(default_value)

    def __get_value(self):

        # make sure that the value set is allowed
        if self._allowed_values is not None:
            assert self._current_value in set(self._allowed_values), 'The value of %s is not in %s' % (self._name, self._allowed_values )

        # construct the class

        out_string = '--%s' % self._name

        if self._is_number:

            out_string += ' %f' % self._current_value

        elif self._is_bool:

            if not self._current_value:

                # we remove the string
                out_string = ''

        else:

            out_string += " '%s'" % self._current_value

        return out_string

    def __set_value(self, value):
        if self._allowed_values is not None:
            assert value in self._allowed_values, 'The value %s of %s is not in %s' % (value,self._name, self._allowed_values)

        self._current_value = value

        self._is_set = True

    value = property(__get_value, __set_value)

    def get_disp_value(self):

        return self._current_value

    @property
    def is_set(self):
        return self._is_set

    @property
    def name(self):
        return self._name

    def display(self):

        print(self._help_string)
        if self._allowed_values is not None:
            print(self._allowed_values)

    # make geetter setter


_required_parameters = [
    'outfile',
    'roi',
    'tstarts',
    'tstops',
    'irf',
    'galactic_model',
    'particle_model',
]

_optional_parameters = [
    'ra', 'dec', 'bin_file', 'tsmin', 'strategy', 'thetamax', 'spectralfiles', 'liketype', 'optimizeposition',
    'datarepository', 'ltcube', 'expomap', 'ulphindex', 'flemin', 'flemax', 'fgl_mode', 'tsmap_spec', 'filter_GTI',
    'likelihood_profile', 'remove_fits_files','log_bins', 'bin_file','source_model'
]


class TransientLATDataBuilder(object):

    def __init__(self, triggername, **init_values):
        """
        Build the command for GtBurst's likelihood analysis 
        and produce the required files for the FermiLATLike 
        plugin

        :param triggername: the trigger name in YYMMDDXXX fermi format
        :returns: 
        :rtype: 

        """

        self._triggername = triggername

        # we create a hash of all the parameters
        # and add them to the class

        # this is a really ugly and long way to do this

        self._parameters = collections.OrderedDict()

        # set the name for this parameter

        name = 'outfile'

        # add it to the hash as a parameter object
        # no value is set UNLESS there is a default

        self._parameters[name] = LATLikelihoodParameter(
            name=name, help_string="File for the results (will be overwritten)", is_number=False)

        # this keeps the user from erasing these objects accidentally

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        # and repeat

        name = 'ra'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, help_string="R.A. of the object (J2000)", is_number=True)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'dec'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, help_string="Dec. of the object (J2000)", is_number=True)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'roi'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, help_string="Radius of the Region Of Interest (ROI)", is_number=True)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'tstarts'

        self._parameters[name] = LATLikelihoodParameter(
                name=name,
                default_value = None,
                help_string="Comma-separated list of start times (with respect to trigger)", 
                is_number=False)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'tstops'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value = None,
            help_string="Comma-separated list of stop times (with respect to trigger)", is_number=False)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'zmax'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, default_value=100., help_string="Zenith cut", is_number=True)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'emin'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, default_value=100., help_string="Minimum energy for the analysis", is_number=True)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'emax'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, default_value=100000., help_string="Maximum energy for the analysis", is_number=True)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'irf'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value='p8_source',
            help_string="Instrument Function to be used (IRF)",
            is_number=False,
            allowed_values=irfs)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'galactic_model'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            help_string="Galactic model for the likelihood",
            is_number=False,
            allowed_values=['template (fixed norm.)', 'template', 'none'])

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'particle_model'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            help_string="Particle model",
            is_number=False,
            allowed_values=['isotr with pow spectrum', 'isotr template', 'none', 'bkge', 'auto'])

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'source_model'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            help_string="Source model",
            default_value='PowerLaw2',
            is_number=False,
            allowed_values=spectra.keys())

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'tsmin'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value=20.,
            help_string="Minimum TS to consider a detection",
            is_number=True,
        )

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'strategy'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value='time',
            help_string="Strategy for Zenith cut: events or time",
            is_number=False,
            allowed_values=['events', 'time'])

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'thetamax'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, default_value=180., help_string="Theta cut", is_number=True)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'spectralfiles'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value='no',
            help_string="Produce spectral files to be used in XSPEC?",
            allowed_values=['yes', 'no'],
            is_number=False)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'liketype'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value='unbinned',
            help_string="Likelihood type (binned or unbinned)",
            allowed_values=['binned', 'unbinned'],
            is_number=False)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'bin_file'

        self._parameters[name] = LATLikelihoodParameter(
                name = name,
                default_value = None,
                help_string = "A string containing a text file 'res.txt start end' will get the start and stop times from the columns 'start' and 'end' in the file res.txt.",
                is_bool = False,
                is_number = False)

        ##################################

        name = 'log_bins'

        self._parameters[name] = LATLikelihoodParameter(
                name = name,
                default_value = None,
                help_string = "Use logarithmically-spaced bins. For example: '1.0 10000.0 30'",
                is_number = False,
                is_bool = False)

        ##################################


        name = 'optimizeposition'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value='no',
            help_string="Optimize position with gtfindsrc?",
            allowed_values=['no', 'yes'],
            is_number=False)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'datarepository'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value=configuration.get('dataRepository'),
            help_string="Dir where data are stored",
            is_number=False)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'ltcube'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, default_value='', help_string="Pre-computed livetime cube", is_number=False)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'expomap'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, default_value='', help_string="Pre-computed exposure map", is_number=False)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'ulphindex'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, default_value=-2, help_string="Photon index for upper limits", is_number=True)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'flemin'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value=100,
            help_string="Lower bound energy for flux/upper limit computation",
            is_number=True)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'flemax'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value=10000,
            help_string="Upper bound energy for flux/upper limit computation",
            is_number=True)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'fgl_mode'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value='fast',
            help_string="Set 'complete' to use all FGL sources, set 'fast' to use only bright sources",
            is_number=False)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'tsmap_spec'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value=None,
            help_string=
            "A TS map specification of the type half_size,n_side. For example: \n 0.5,8' makes a TS map 1 deg x 1 deg with 64 points",
            is_number=False)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'filter_GTI'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value=False,
            help_string="Automatically divide time intervals crossing GTIs",
            is_bool=True,
            is_number=False)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'likelihood_profile'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value=False,
            help_string="Produce a text file containing the profile of the likelihood for a \n changing normalization ",
            is_bool=True,
            is_number=False)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        ##################################

        name = 'remove_fits_files'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value=False,
            help_string="Whether to remove the FITS files of every interval in order to save disk space",
            is_bool=True,
            is_number=False)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        # Now if there are keywords from a configuration to read,
        # lets do it

        self._process_keywords(**init_values)

    def _process_keywords(self, **kwargs):
        """
        processes the keywords from a dictionary 
        likely loaded from a yaml config

        :returns: 
        :rtype: 

        """

        for k, v in kwargs.items():

            if k in self._parameters:

                self._parameters[k].value = v

            else:
                # add warning that there is something strange in the configuration
                pass
 
    def __setattr__(self, name, value):
        """
        Override this so that we cannot erase parameters
        
        """

        if (name in _required_parameters) or (name in _optional_parameters):
            raise AttributeError("%s is an immutable attribute." % name)
        else:

            super(TransientLATDataBuilder, self).__setattr__(name, value)

    def _get_command_string(self):
        """
        This builds the cmd string for the script
        """
        executable = os.path.join('fermitools', 'GtBurst', 'scripts', 'doTimeResolvedLike.py')
        cmd_str = '%s %s' % (executable,
                             self._triggername)

        for k, v in self._parameters.items():

            # only add on the parameters that are set

            if v.is_set:

                cmd_str += ' %s' % v.value

            else:

                # but fail if we did not set the ones needed

                assert v.name not in _required_parameters, '%s is not set but is required' % v.name

        return cmd_str

    def run(self, include_previous_intervals=False, recompute_intervals=False):
        """
        run GtBurst to produce the files needed for the FermiLATLike plugin
        """

        assert has_fermitools, 'You do not have the fermitools installed and cannot run GtBurst'

        # This is not the cleanest way to do this, but at the moment I see
        # no way around it as I do not want to rewrite the fermitools

        cmd = self._get_command_string()    # should not allow you to be missing args!

        # now we want to get the site package directory to find where the script is
        # located. This should be the first entry... might break in teh future!

        site_pkg = site.getsitepackages()[0]

        cmd = os.path.join(site_pkg, cmd)
        executable   = cmd.split()[0]
        gtapp_mp_dir = os.path.join(site_pkg,'fermitools', 'GtBurst', 'gtapps_mp')
        executables  = [
            executable,
            os.path.join(gtapp_mp_dir, 'gtdiffrsp_mp.py'),
            os.path.join(gtapp_mp_dir, 'gtexpmap_mp.py'),
            os.path.join(gtapp_mp_dir, 'gtltcube_mp.py'),
            os.path.join(gtapp_mp_dir, 'gttsmap_mp.py'),
        ]
        for _e in executables:
            print ("Changing permission to %s" % _e)
            os.chmod(_e, 0o755)

        log.info('About to run the following command:\n%s' % cmd)

        # see what we already have

        intervals_before_run = glob('interval*-*')

        if recompute_intervals:

            if intervals_before_run:

                # ok, perhaps the user wants to recompute intervals in this folder.
                #  We will move the old intervals to a tmp directory
                # so that they are not lost

                log.info('You have choosen to recompute the time intervals in this folder')

                tmp_dir = 'tmp_%s' % str(uuid.uuid4())

                log.info('The older entries will be moved to %s' % tmp_dir)

                os.mkdir(tmp_dir)

                for interval in intervals_before_run:

                    shutil.move(interval, tmp_dir)

                # now remove these
                intervals_before_run = []

        if include_previous_intervals:

            intervals_before_run = []

        # run this baby

        subprocess.call(cmd, shell=True)
        self.lat_observations = self._create_lat_observations_from_run(intervals_before_run)
        return self.lat_observations

    def _create_lat_observations_from_run(self, intervals_before_run):
        """
        After a run of gtburst, this collects the all the relevant files from
        each inteval and turns them into LAT observations.


        :rtype: 

        """

        # scroll thru the intervals that were created

        # place them in LAT observations

        # attach them to dictionary

        # dir = the interval

        lat_observations = []

        # need a strategy to collect the intervals
        intervals = sorted(glob('interval*-*'))

        for interval in intervals:

            if interval in intervals_before_run:

                log.info(
                    '%s existed before this run,\n it will not be auto included in the list,\n but you can manually see grab the data.' % interval
                )
            else:

                # check that either tstarts,tstops or log_bins are defined
                if self._parameters['tstarts'].value is not None or self._parameters['tstops'].value is not None:
                    tstart, tstop = [
                        float(x) for x in re.match('^interval(-?\d*\.\d*)-(-?\d*\.\d*)\/?$', interval).groups()]
                else:
                    assert self._parameters['log_bins'].value is None, 'Choose either to use tstarts and tstops, or to use log_bins'

                event_file = os.path.join(interval, 'gll_ft1_tr_bn%s_v00_filt.fit' % self._triggername)

                if not file_existing_and_readable(event_file):
                    log.info('The event file does not exist. Please examine!')

                ft2_file = os.path.join(interval, 'gll_ft2_tr_bn%s_v00_filt.fit' % self._triggername)

                if not file_existing_and_readable(ft2_file):
                    log.info('The ft2 file does not exist. Please examine!')
                    log.info('we will grab the data file for you.')

                    base_ft2_file = os.path.join('%s' % self.datarepository.get_disp_value(),
                                                 'bn%s' % self._triggername,
                                                 'gll_ft2_tr_bn%s_v00.fit' % self._triggername)

                    if not file_existing_and_readable(base_ft2_file):

                        log.error('Cannot find any FT2 files!')

                        raise AssertionError()
                        
                    shutil.copy(base_ft2_file, interval)

                    ft2_file = os.path.join(interval, 'gll_ft2_tr_bn%s_v00.fit' % self._triggername)

                    log.info('copied %s to %s' % (base_ft2_file, ft2_file))

                exposure_map = os.path.join(interval, 'gll_ft1_tr_bn%s_v00_filt_expomap.fit' % self._triggername)

                if not file_existing_and_readable(exposure_map):
                    log.info('The exposure map does not exist. Please examine!')

                livetime_cube = os.path.join(interval, 'gll_ft1_tr_bn%s_v00_filt_ltcube.fit' % self._triggername)

                if not file_existing_and_readable(livetime_cube):
                    log.info('The livetime_cube does not exist. Please examine!')

                # optional bin_file parameter
                #if self._parameters['bin_file'].value is not None:
                    
                    # liketype matches
                #    assert self._parameters['liketype'] == 'binned', 'liketype must be binned to use bin_file parameter %s'%self._parameters['bin_file'].value

                    # value carries a few arguments, take first value- path
                 #   bin_file_path = self._parameters['bin_file'].value.split()[0]
                 #   bin_file = os.path.join(interval, bin_file_path)

                 #   if not file_existing_and_readable(bin_file):
                 #       print('The bin_file at %s does not exist. Please examine!'%bin_file)

                # now create a LAT observation object
                this_obs = LATObservation(event_file, ft2_file, exposure_map, livetime_cube, tstart, tstop, self._parameters['liketype'].get_disp_value(), self._triggername)#@, bin_file)

                lat_observations.append(this_obs)

        return lat_observations

    def to_LATLike(self):

        _lat_like_plugins=[]

        for _lat_ob in self.lat_observations:

            _lat_like_plugins.append(_lat_ob.to_LATLike())

        return _lat_like_plugins

    def display(self, get=False):
        """
        Display the current set parameters
        """

        out = collections.OrderedDict()

        for k, v in self._parameters.items():

            if v.is_set:

                out[k] = v.get_disp_value()

        df = pd.Series(out)

        print(df)

        if get:

            return df

    def __repr__(self):

        return self.display(get=True).to_string()

    def save_configuration(self, filename):
        """
        Save the current configuration to a yaml 
        file for use later. Suggested extension is .yml

        :param filename: the yaml file name to save to 
        :returns: 
        :rtype: 

        """

        # create a temporary dict to hold
        # the set values

        data = {}

        for k, v in self._parameters.items():

            if v.is_set:

                data[k] = v.get_disp_value()

        with open(filename, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    @classmethod
    def from_saved_configuration(cls, triggername, config_file):
        """
        Load a saved yaml configuration for the given trigger name

        :param triggername: Trigger name of the source in YYMMDDXXX 
        :param config_file: the saved yaml configuration to use
        :returns: 
        :rtype: 

        """

        with open(config_file, 'r') as stream:
            loaded_config = yaml.safe_load(stream)

        return cls(triggername, **loaded_config)


class LATObservation(object):

    def __init__(self, event_file, ft2_file, exposure_map, livetime_cube, tstart, tstop, liketype, triggername):#, bin_file):
        """
        A container to formalize the storage of Fermi LAT 
        observation files

        :param event_file: 
        :param ft2_file: 
        :param exposure_map: 
        :param livetime_cube: 
        :param tstart:
        :param tstop:
        :param liketype:
        :param triggername:
        :returns: 
        :rtype: 

        """

        self._event_file = event_file
        self._ft2_file = ft2_file
        self._exposure_map = exposure_map
        self._livetime_cube = livetime_cube
        self._tstart = tstart
        self._tstop = tstop
        self._liketype =liketype
        self._triggername = triggername
        #self._bin_file = bin_file

    @property
    def event_file(self):
        return self._event_file

    @property
    def ft2_file(self):
        return self._ft2_file

    @property
    def exposure_map(self):
        return self._exposure_map

    @property
    def livetime_cube(self):
        return self._livetime_cube

    @property
    def tstart(self):
        return self._tstart

    @property
    def tstop(self):
        return self._tstop

    @property
    def liketype(self):
        return self._liketype

    #@property
    #def bin_file(self):
    #    return self._bin_file

    @property
    def triggername(self):
        return self._triggername

    def __repr__(self):

        output = collections.OrderedDict()

        output['time interval'] = '%.3f-%.3f' % (self._tstart, self._tstop)
        output['event_file'] = self._event_file
        output['ft2_file'] = self._ft2_file
        output['exposure_map'] = self._exposure_map
        output['livetime_cube'] = self._livetime_cube
        output['triggername'] = self._triggername
        output['liketype'] = self._liketype
        #output['bin_file'] = self._bin_file

        df = pd.Series(output)

        return df.to_string()

    def to_LATLike(self):

        _fermi_lat_like =  FermiLATLike(
                     name = ('LAT%dX%d' % (self._tstart, self._tstop)).replace('-','n'),
                     event_file = self._event_file,
                     ft2_file = self._ft2_file,
                     livetime_cube_file = self._livetime_cube,
                     kind = self._liketype,
                     exposure_map_file=self._exposure_map,
                     source_maps=None,
                     binned_expo_map=None)
                    #,bin_file = self._bin_file)
                     #source_name=self._triggername)

        return _fermi_lat_like
