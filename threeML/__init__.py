import os
import sys
import glob
import inspect
import imp

#This dynamically loads a module and return it in a variable
def __import__(name, globals=None, locals=None, fromlist=None):
  # Fast path: see if the module has already been imported.
  try:
      return sys.modules[name]
  except KeyError:
      pass

  # If any of the following calls raises an exception,
  # there's a problem we can't handle -- let the caller handle it.

  fp, pathname, description = imp.find_module(name)

  try:
      return imp.load_module(name, fp, pathname, description)
  except:
      raise
  finally:
      # Since we may exit via an exception, close fp explicitly.
      if fp:
          fp.close()
  pass
pass

#Find the directory containing 3ML
threeML_dir                   = os.path.abspath(os.path.dirname(__file__))

#Import all modules here
sys.path.insert(0,threeML_dir)
mods                          = [ os.path.basename(f)[:-3] for f in glob.glob(os.path.join(threeML_dir,"*.py"))]
#Filter out __init__
modsToImport                  = filter(lambda x:x.find("__init__")<0,mods)

#Import everything in current directory
for mod in modsToImport:
  exec("from %s import *" %(mod))


#Now look for plugins
plugins_dir                   = os.path.join(os.path.dirname(__file__),"plugins")
sys.path.insert(1,plugins_dir)
mplugins                      = glob.glob(os.path.join(plugins_dir,"*.py"))
#Filter out __init__
mplugins                      = filter(lambda x:x.find("__init__")<0,mplugins)

msgs                          = []

for i,plug in enumerate(mplugins):
  #Loop over each candidates plugins
  try:
    thisPlugin                  = __import__(os.path.basename(".".join(plug.split(".")[:-1])))
  except:
    raise
    print("\nWARNING: Could not import plugin %s. Do you have the relative instrument software installed and configured?" %(plug))
    continue
  #Get the classes within this module
  classes                     = inspect.getmembers(thisPlugin,lambda x:inspect.isclass(x) and inspect.getmodule(x)==thisPlugin)
  for name,cls in classes:
    if(not issubclass(cls,pluginPrototype)):
      #This is not a plugin
      pass
    else:
      string                  = "%s for %s" %(cls.__name__,thisPlugin.__instrument_name)
      msgs.append("* %-60s available" %(string))
      #import it again in the uppermost namespace
      exec("from %s import %s" %(os.path.basename(thisPlugin.__name__),cls.__name__))
    pass
  pass
pass

def getAvailablePlugins():
  print("Plugins:\n")
  print("\n".join(msgs))
