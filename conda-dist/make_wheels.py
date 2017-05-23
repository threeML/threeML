import subprocess
import logging
import glob
import contextlib
import os
import shutil

# This is a list of packages that we do not want to generate wheels for
# (they are already in the main conda repository or we have custom recipes)
excluded_packages = ['subprocess32', 'functools32', 'algopy',
                     'scipy','numpy','pandas', 'astropy', 'numexpr',
                     'appdirs', 'cycler', 'dill',
                     'iminuit', 'matplotlib', 'pyparsing', 'pytz',
                     'PyYAML', 'python_dateutil', 'requests', 'setuptools',
                     'six', 'tables', 'packaging', 'uncertainties', 'astromodels']

def is_excluded(package_name):
    
    for excl in excluded_packages:
        
        if excl in package_name:
            
            return True
    
    return False
    
    

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


@contextlib.contextmanager
def within_directory(dirname):
    
    cur_dir = os.getcwd()
    
    os.chdir(dirname)
    
    yield
    
    os.chdir(cur_dir)
    

def run_command(command_line):
    
    logging.info("About to run:")
    logging.info(command_line)
    
    subprocess.check_call(command_line, shell=True)

# Get current directory
root_dir = os.path.abspath(os.getcwd())

# Download 3ML from the github

cmdline = 'pip download git+https://github.com/giacomov/3ML.git'
run_command(cmdline)

# Remove all wheels for packages to be excluded
wheels = glob.glob("*.whl")

for wheel in wheels:
    
    package_name = wheel.split("-")[0]
    
    if is_excluded(package_name):
        
        os.remove(wheel)

# Loop over all packages that do not have a wheel
tars = glob.glob("*.tar.gz")
zips = glob.glob("*.zip")

archives = tars + zips

pkg_type = None

for tar in archives:
    
    # Get the name of the package
    if '.tar.gz' in tar:
    
        package_name = os.path.basename(tar).split(".tar.gz")[0]
        
        pkg_type = 'tar'
        
    elif '.zip' in tar:
        
        package_name = os.path.basename(tar).split(".zip")[0]
        
        pkg_type = 'zip'
    
    else:
        
        raise RuntimeError("Should never get here")
    
    if is_excluded(package_name):
        
        logging.info("Package %s is excluded" % package_name)
        
        os.remove(tar)
        
        continue
    
    # See if we already have the wheel (from a previous run)
    wheels = glob.glob("%s*.whl" % package_name)
    
    if len(wheels) > 0:
        
        logging.info("Already have wheel %s for %s" % (wheels[0], package_name))
    
    logging.info("Processing %s" % package_name)
    
    if pkg_type == 'tar':
        
        cmdline = "tar zxvf %s" % tar
        run_command(cmdline)
    
    else:
        
        cmdline = 'unzip %s' % tar
        run_command(cmdline)
    
    if not os.path.exists(package_name):
        
        # try without version
        package_name = package_name.split("-")[0]
    
    # Go into the directory
    with within_directory(package_name):
        
        cmdline = 'python setup.py bdist_wheel'
        
        run_command(cmdline)
        
        # Now find the wheel and move it to the main directory
        wheels = glob.glob("./dist/%s-*.whl" % package_name)
        
        assert len(wheels)==1
        
        logging.info("Moving wheel %s to %s" % (wheels[0], root_dir))
        
        shutil.move(wheels[0], root_dir)
    
    # Remove the directory
    
    shutil.rmtree(package_name)
    
    # Remove the archive
    os.remove(tar)
        
