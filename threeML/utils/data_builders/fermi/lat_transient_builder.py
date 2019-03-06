import collections

import pandas as pd
import site
import subprocess

try:
    from GtBurst import IRFS
    irfs = IRFS.IRFS.keys()
    irfs.append('auto')

    has_fermitools = True

except(ImportError):

    has_fermitools = False



class LATLikelihoodParameter(object):

    def __init__(self, name, help_string, default_value=None, allowed_values=None, is_number=True):
        """

        A container for the parameters that are needed by GtBurst

        :param name: 
        :param help_string: 
        :param default_value: 
        :param allowed_values: 
        :param is_number: 
        :returns: 
        :rtype: 

        """
       

        self._name = name
        self._allowed_values = allowed_values
        self._default_value = default_value
        self._is_number = is_number
        self._help_string = help_string
        self._is_set = False

        # if there is a default value, lets go ahead and set it
        
        if default_value is not None:
            self.__set_value(default_value)

    def __get_value(self):
        

        # make sure that the value set is allowed
        if self._allowed_values is not None:
            assert self._current_value in self._allowed_values, 'The value of %s is not in %s' % (self._name, 'test')

        # construct the class
            
        out_string = '--%s' % self._name

        if self._is_number:

            out_string += ' %f' % self._current_value

        else:

            out_string += ' %s' % self._current_value

        return out_string

    def __set_value(self, value):
        if self._allowed_values is not None:
            assert value in self._allowed_values, 'The value of %s is not in %s' % (self._name, 'test')

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
    'log_bins',
    'irf',
    'galactic_model',
    'particle_model',
]

_optional_parameters = [
    'ra', 'dec', 'bin_file', 'tsmin', 'strategy', 'thetamax', 'spectralfiles', 'liketype', 'optimizeposition',
    'datarepository', 'ltcube', 'expomap', 'ulphindex', 'flemin', 'flemax', 'fgl_mode', 'tsmap_spec', 'filter_GTI',
    'likelihood_profile', 'remove_fits_files'
]


class TransientLATDataBuilder(object):

    def __init__(self, triggername, **init_values):
        """
        Build the command for GtBurst's likelihood analysis 
        and produce the required files for the FermiLATLike 
        plugin

        :param triggername: 
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

        name = 'dec'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, help_string="Dec. of the object (J2000)", is_number=True)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        name = 'roi'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, help_string="Radius of the Region Of Interest (ROI)", is_number=True)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        name = 'tstarts'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, help_string="Comma-separated list of start times (with respect to trigger)", is_number=False)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        name = 'tstops'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, help_string="Comma-separated list of stop times (with respect to trigger)", is_number=False)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        name = 'zmax'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, default_value=100., help_string="Zenith cut", is_number=True)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        name = 'emin'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, default_value=100., help_string="Minimum energy for the analysis", is_number=True)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        name = 'emax'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, default_value=100000., help_string="Maximum energy for the analysis", is_number=True)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        name = 'irf'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value='auto',
            help_string="Instrument Function to be used (IRF)",
            is_number=False,
            allowed_values=irfs)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        name = 'galactic_model'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            help_string="Galactic model for the likelihood",
            is_number=False,
            allowed_values=['template (fixed norm.)', 'template', 'none'])

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        name = 'particle_model'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            help_string="Particle model",
            is_number=False,
            allowed_values=['auto', 'isotr with pow spectrum', 'isotr template', 'none', 'bkge'])

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        name = 'tsmin'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value=20.,
            help_string="Minimum TS to consider a detection",
            is_number=True,
        )

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        name = 'strategy'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value='time',
            help_string="Strategy for Zenith cut: events or time",
            is_number=False,
            allowed_values=['events', 'time'])

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        name = 'thetamax'

        self._parameters[name] = LATLikelihoodParameter(
            name=name, default_value=180., help_string="Theta cut", is_number=True)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])

        name = 'spectralfiles'

        self._parameters[name] = LATLikelihoodParameter(
            name=name,
            default_value='no',
            help_string="Produce spectral files to be used in XSPEC?",
            allowed_values=['yes', 'no'],
            is_number=False)

        super(TransientLATDataBuilder, self).__setattr__(name, self._parameters[name])


        
    def __setattr__(self, name, value):
        """
        OVerride this so that we cannot erase parameters
        
        """
        
        if (name in _required_parameters) or (name in _optional_parameters):
            raise AttributeError("%s is an immutable attribute." % name)
        else:

            super(TransientLATDataBuilder, self).__setattr__(name, value)

    def _get_command_string(self):
        """
        This builds the cmd string for the script
        """
        

        cmd_str = '%s %s' % (os.path.join('fermitools','GtBurst','scripts','doTimeResolvedLike.py'), self._triggername)

        for k, v in self._parameters.items():

            # only add on the parameters that are set
            
            if v.is_set:

                cmd_str += ' %s' % v.value

            else:

                # but fail if we did not set the ones needed

                assert v.name not in _required_parameters, '%s is not set but is required' % v.name

        return cmd_str

    def run(self):
        """
        run GtBurst to produce the files needed for the FermiLATLike plugin
        """

        # This is not the cleanest way to do this, but at the moment I see
        # no way around it as I do not want to rewrite the fermitools

        cmd = self._get_command_string() # should not allow you to be missing args!

        # now we want to get the site package directory to find where the script is
        # located. This should be the first entry... might break in teh future!

        site_pkg = site.getsitepackages()[0]

        subprocess.call(os.path.join(site_pkg,cmd), shell=True)
        

    
    def display(self):
        """
        Display the currently set parameters
        """

        out = collections.OrderedDict()

        for k, v in self._parameters.items():

            if v.is_set:

                out[k] = v.get_disp_value()

        df = pd.Series(out)

        print(df)




