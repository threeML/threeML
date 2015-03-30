#!/usr/bin/python
# angsep.py
# Program to calculate the angular separation between two points
# whose coordinates are given in RA and Dec
# From angsep.py Written by Enno Middelberg 2001

from numpy import *
import string
import sys

def angsep(ra1deg,dec1deg,ra2deg,dec2deg):
    """ Determine separation in degrees between two celestial objects
        arguments are RA and Dec in decimal degrees.
        """
    ra1rad=ra1deg*pi/180
    dec1rad=dec1deg*pi/180
    ra2rad=ra2deg*pi/180
    dec2rad=dec2deg*pi/180
    
    # calculate scalar product for determination
    # of angular separation
    
    x=cos(ra1rad)*cos(dec1rad)*cos(ra2rad)*cos(dec2rad)
    y=sin(ra1rad)*cos(dec1rad)*sin(ra2rad)*cos(dec2rad)
    z=sin(dec1rad)*sin(dec2rad)
    
    rad=arccos(x+y+z) # Sometimes gives warnings when coords match
    
    # use Pythargoras approximation if rad < 1 arcsec
    sep = choose( rad<0.000004848 , (sqrt((cos(dec1rad)*(ra1rad-ra2rad))**2+(dec1rad-dec2rad)**2),rad))
        
    # Angular separation
    sep=sep*180/pi
                                     
    return sep