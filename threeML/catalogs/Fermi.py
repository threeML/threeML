import numpy

from VirtualObservatoryCatalog import VirtualObservatoryCatalog

class GBMBurstCatalog(VirtualObservatoryCatalog):
    
    def __init__(self):
        
        super(GBMBurstCatalog, self).__init__('fermigbrst', 
                           'http://heasarc.gsfc.nasa.gov/cgi-bin/vo/cone/coneGet.pl?table=fermigbrst&',
                           'Fermi/GBM burst catalog')

    def applyFormat(self, votable):
        
        table = votable.to_table() 
        table['ra'].format = '5.3f'
        table['dec'].format = '5.3f'
        return table

#########

threefgl_types = {
'agn' : 'other non-blazar active galaxy',
'bcu' : 'active galaxy of uncertain type',
'bin' : 'binary',
'bll' : 'BL Lac type of blazar',
'css' : 'compact steep spectrum quasar',
'fsrq' : 'FSRQ type of blazar',
'gal' : 'normal galaxy (or part)',
'glc' : 'globular cluster',
'hmb' : 'high-mass binary',
'nlsy1' : 'narrow line Seyfert 1',
'nov' : 'nova',
'PSR' : 'pulsar, identified by pulsations',
'psr' : 'pulsar, no pulsations seen in LAT yet',
'pwn' : 'pulsar wind nebula',
'rdg' : 'radio galaxy',
'sbg' : 'starburst galaxy',
'sey' : 'Seyfert galaxy',
'sfr' : 'star-forming region',
'snr' : 'supernova remnant',
'spp' : 'special case - potential association with SNR or PWN',
'ssrq' : 'soft spectrum radio quasar',
'' : 'unknown'
}

class LATSourceCatalog(VirtualObservatoryCatalog):
    
    def __init__(self):
        
        super(LATSourceCatalog, self).__init__('fermilpsc', 
                           'http://heasarc.gsfc.nasa.gov/cgi-bin/vo/cone/coneGet.pl?table=fermilpsc&',
                           'Fermi/LAT source catalog')        

    def applyFormat(self, votable):
        
        table = votable.to_table() 
        table['ra'].format = '5.3f'
        table['dec'].format = '5.3f'
        table['Search_Offset'].format = '5.3f'
        
        def translate(key):
            if(key.lower()=='psr'):
                return threefgl_types[key]
            else:
                return threefgl_types[key.lower()]
        
        #Translate the 3 letter code to a more informative category, according
        #to the dictionary above       
        
        table['source_type'] = numpy.array(map(translate, table['source_type']))
                                
        new_table = table['name',
                          'source_type',
                          'ra','dec',
                          'assoc_name_1',
                          'tevcat_assoc',
                          'Search_Offset']
                
        return new_table.group_by('Search_Offset')
 
