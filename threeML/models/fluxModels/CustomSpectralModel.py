from threeML.models.spectralmodel import SpectralModel

from threeML.models.Parameter import Parameter

import re, parse, parser

import collections

import numpy

class CustomSpectralModel( SpectralModel ):
    
    def setup(self, modelDef, **kwargs):
        
        if 'function' in kwargs.keys():
            
            self.userFunction = kwargs['function']
        
        else:
            
            self.userFunction = None
        
        #Divide in lines
        lines = modelDef.replace("\r","\n").split("\n")
        
        #Remove empty lines
        
        filtered = filter(lambda x: not re.match(r'^\s*$', x), lines)
        
        if len(filtered) <= 2:
            
            raise RuntimeError("You have to provide at least a name:formula line and a init line.")
        
        #Parse first line: model name and formula
        
        r = parse.parse('{name}:{formula}', filtered[0].replace(" ",""))
        
        if r is None:
            
            raise RuntimeError("Could not understand first line: %s" %filtered[0])
        
        self.functionName = r['name']
        self.formula = r['formula']
        
        #Second line: init values
        
        r = parse.parse('init:{initValues}', filtered[1].replace(" ",""))
        
        if r is None:
            
            raise RuntimeError("Could not understand second line: %s" %filtered[1])
        
        paramNames = []
        initValues = []
        
        for pp in r['initValues'].split(","):
            
            try:
            
                name,value = pp.split("=")
            
            except:
                
                raise RuntimeError("Cannot understand init value %s" % pp)
            
            if name=='e' or name=='E':
                
                raise RuntimeError("You cannot use 'e' or 'E' as parameter names. They indicate energy.")
            
            paramNames.append( name )
            initValues.append( float(value) )
        
        #Make the parameters dictionary
        
        self.parameters = collections.OrderedDict()
        
        for p,v in zip( paramNames, initValues ):
            
            #If the name is all upper case, it is a scale or normalization
            #parameter
            if p.upper()==p:
                
                norm = True
            
            else:
                
                norm = False
            
            self.parameters[p] = Parameter(p, v, None, None, v / 10.0, normalization=norm)
        
        #Third line: boundaries
        
        if len(filtered) > 2:
        
            r = parse.parse('bounds:{boundsSpec}', filtered[2].replace(" ",""))
            
            if r is None:
                
                raise RuntimeError("Could not understand third line: %s" %(filtered[2]))
            
            boundsExpr = r['boundsSpec'].split(",")
            
            for p in self.parameters.keys():
                
                #Find out if there is a bound specification for this parameter
                thisExpr = filter(lambda x:x.find(p) >= 0, boundsExpr)
                
                if len(thisExpr)==0:
                    
                    #No, just continue
                    continue
                
                elif len(thisExpr) > 1:
                    
                    raise RuntimeError("More than one bounds specification for parameter %s" % p)
                
                #If we get here, there is a specification for the bounds
                #Parse it
                pexpr = thisExpr[0]
                
                #Remove it from the boundsExpr list
                boundsExpr.pop( boundsExpr.index(pexpr) )
                
                r = parse.parse('{minvalue}<={pname}<={maxvalue}', pexpr)
            
                if r is not None:
                
                    #Max and min specified
                    minValue = float( r['minvalue'] )
                    maxValue = float( r['maxvalue'] )
                
                else:
                    
                    r = parse.parse('{pname}<={maxvalue}',pexpr)
                
                    if r is not None:
                        
                        minValue = None
                        maxValue = float( r['maxvalue'] )
                    
                    else:
                        
                        r = parse.parse('{pname}>={minvalue}', pexpr)
                        
                        if r is not None:
                            
                            minValue = float( r['minvalue'] )
                            maxValue = None
                        
                        else:
                            
                            raise RuntimeError("Could not understand expression %s" %pexpr)
                
                self.parameters[p].setBounds(minValue,maxValue)
            
            #If the boundsExpr is not empty at this point, it means that there is a bounds
            #specification for a parameter which does not exists...
            if len(boundsExpr) > 0:
                
                raise RuntimeError("One or more bounds refer to non-existent parameters.")
            
        pass #End of boundaries parsing
        
        #Now sanitize the formula removing spaces
        
        self.formula = "".join( self.formula.split() )
        
        self.finalSetup()
        
    
    def __getstate__( self ):
        
        #This is used by pickle to get the state of the current
        #instance. The return value of this method will be used
        #by the __setstate__ method on the other side (the remote
        #side) to recreate the class
        
        d = {}
        
        d['formula'] = self.formula
        d['userFunction'] = self.userFunction
        d['parameters'] = self.parameters
        
        return d
    
    def __setstate__( self, state ):
        
        self.formula = state['formula']
        self.userFunction = state['userFunction']
        self.parameters = state['parameters']
        
        self.finalSetup()
        
    def finalSetup( self ):
        
        #Now compile the formula
        
        self._compiledFormula = parser.expr(self.formula).compile()
        
        #Create a local dictionary to store variables which will be used
        #by the formula just compiled
        self._locals = {}
        
        #Create pointers to normal numpy functions, so that they can be used
        #in the formula
        self._locals['log'] = numpy.log
        self._locals['log10'] = numpy.log10
        self._locals['power'] = numpy.power
        self._locals['exp'] = numpy.exp
        
        #If the user specified a user-defined formula, add it to the
        #dictionary
        
        if not self.userFunction is None:
            
            fname = self.userFunction.__name__
            
            self._locals[fname] = self.userFunction
        
        #Test it with default parameters
        
        e = numpy.linspace(1.0,100,10)
        
        try:
            
            testValue = self.__call__( e )
        
        except:
                        
            print("\n\nTest did not work. Check your syntax (Remember: parameters are case-sensitive!).\n\n")
            
            raise
        
        else:
            
            if numpy.sum( ~numpy.isfinite(testValue) )==0:
                
                #Everything seems to be fine
                
                pass
            
            else:
                
                raise RuntimeError("The compiled function returned NaN or inf. Check your syntax.")
    
    def __call__(self, e):
                    
        for p in self.parameters.keys():
                
            self._locals[p] = self.parameters[p].value
            
        self._locals['e'] = numpy.array( e )
            
        return eval( self._compiledFormula, self._locals )
