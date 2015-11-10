import numpy

class Binner( object ):
    
    def __init__( self, edges1, edges2, counts ):
        
        self.counts = numpy.array( counts )
        
        self.e1 = numpy.array( edges1 )
        self.e2 = numpy.array( edges2 )
        
        self.total = numpy.sum( self.counts )
    
    def byConstantCounts( self, n, secondVector = None ):
        '''
        Bin data to have an equal number of counts in each bin
        '''
        
        #Integral distribution
        
        intDistr = numpy.cumsum( self.counts )
        
        if secondVector is not None:
            
            intDistr2 = numpy.cumsum( secondVector )
        
        #Get the indexes of the elements in intDistr
        #where the value changes by more than n
        
        idx = []
        values = []
        newV = []
        
        end = []
        
        reference = 0
        
        ref2 = 0
        
        for i, d in enumerate( intDistr ):
            
            value = d - reference
            
            if value >= n:
                
                idx.append( i )
                
                values.append( value )
                
                reference = d
                
                end.append( self.e2[i] )
                
                if secondVector is not None:
                    
                    newV.append( intDistr2[i] - ref2 )
                    
                    ref2 = intDistr2[i]
                
                
        #Always add the last point
        if self.counts.shape[0] - 1 not in idx:
            
            idx.append( self.counts.shape[0] - 1)
            
            values.append( self.total - reference )
            
            if secondVector is not None:
                
                newV.append( numpy.sum(secondVector) - ref2 )
            
            end.append( self.e2[-1] )
        
        #Now create the start of the edges
        
        start = [ self.e1[0] ]
        start.extend( end[:-1] )
        
        #Safety check
        
        if numpy.sum( values ) != self.total:
            
            raise RuntimeError("This is a bug in the Binner. The total counts are not the same as the input")
        
        return numpy.array( start ), numpy.array( end ), numpy.array( values ), numpy.array( newV )
