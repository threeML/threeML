# Author: L. Tibaldo, ltibaldo@slac.stanford.edu

from threeML.models.spatialmodel import SpatialModel
from threeML.models.Parameter import Parameter, SpatialParameter
import numpy as np
import collections
from angsep import angsep


class Disk(SpatialModel):
    def setup(self):
        self.functionName = "Disk"
        self.formula = r'$f({\rm RA, Dec}) = \left(\frac{180^\circ}{\pi}\right)^2 \frac{1}{\pi r^2} \, \left({\rm angsep} ({\rm RA, Dec, RA_0, Dec_0}) < r \right) $'
        self.parameters = collections.OrderedDict()
        self.parameters['RA0'] = Parameter('RA0', 1., 0, 360, 0.1, fixed=False, nuisance=False, dataset=None)
        self.parameters['Dec0'] = Parameter('Dec0', 1., -90, 90, 0.1, fixed=False, nuisance=False, dataset=None)
        self.parameters['radius'] = SpatialParameter('radius', 0.1, 0, 20, 0.01, fixed=False, nuisance=False,
                                                     dataset=None)

        self.ncalls = 0

    def __call__(self, RA, Dec, energy):
        self.ncalls += 1
        RA0 = self.parameters['RA0'].value
        Dec0 = self.parameters['Dec0'].value
        radius = self.parameters['radius'].getValue(energy)

        return np.power(180 / np.pi, 2) * 1. / (np.pi * radius ** 2) * (angsep(RA, Dec, RA0, Dec0) < radius)

    def integratedFlux(self, energy):
        return 1.

    def getBoundaries(self):

        maxRadius = self.parameters['radius'].maxValue

        minDec = max(-90.,self.parameters['Dec0'].value - maxRadius)
        maxDec = min(90.,self.parameters['Dec0'].value + maxRadius)

        maxAbsDec = max(np.absolute(minDec),np.absolute(maxDec))
        if maxAbsDec > 89. or maxRadius/np.cos(maxAbsDec*np.pi/180.) >= 180.:
            minRa = 0.
            maxRa = 360.
        else:
            minRa = self.parameters['RA0'].value - maxRadius/np.cos(maxAbsDec*np.pi/180.)
            maxRa = self.parameters['RA0'].value + maxRadius/np.cos(maxAbsDec*np.pi/180.)
            if minRa < 0.:
                minRa = minRa + 360.
            elif maxRa > 360.:
                maxRa = maxRa - 360.

        return minRa, maxRa, minDec, maxDec

