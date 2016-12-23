#Author: Giacomo Vianello (giacomov@slac.stanford.edu)

#A hack on the astropy Table class to make its output
#more appealing, especially when in the Ipython notebook

import pandas as pd
import astropy.table


def long_path_formatter(line, max_width=pd.get_option('max_colwidth')):
    """
    If a path is longer than max_width, it substitute it with the first and last element,
    joined by "...". For example 'this.is.a.long.path.which.we.want.to.shorten' becomes
    'this...shorten'

    :param line:
    :param max_width:
    :return:
    """

    if len(line) > max_width:

        tokens = line.split(".")
        trial1 = "%s...%s" % (tokens[0], tokens[-1])

        if len(trial1) > max_width:

            return "...%s" %(tokens[-1][-1:-(max_width-3)])

        else:

            return trial1

    else:

        return line


class Table(astropy.table.Table):
    
    def _base_repr_(self, html=False, show_name=True, **kwargs):
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
    
    def _base_repr_(self, html=False,**kwargs):
        return super(NumericMatrix, self)._base_repr_(html, show_name=False, **kwargs)
    
