#Author: Giacomo Vianello (giacomov@slac.stanford.edu)

#A hack on the astropy Table class to make its output
#more appealing, especially when in the Ipython notebook

import astropy.table
from astropy.utils.xml.writer import xml_escape

class Table(astropy.table.Table):
    
    def _base_repr_(self, html=False, show_name=True):
        '''Override the method in the astropy.Table class
        to avoid displaying the description, and the format
        of the columns'''
        
        tableid = 'table{id}'.format(id=id(self))
        
        data_lines, outs = self.formatter._pformat_table(self,
            tableid=tableid, html=html, max_width=(-1 if html else None),
            show_name=show_name, show_unit=None, show_dtype=False)
                
        out = '\n'.join(data_lines)
        if astropy.table.six.PY2 and isinstance(out, astropy.table.six.text_type):
            out = out.encode('utf-8')

        return out

class NumericMatrix(Table):
    
    def _base_repr_(self, html=False):
        return super(NumericMatrix, self)._base_repr_(html, False)
    
