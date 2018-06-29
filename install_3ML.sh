#!/bin/bash

# Process options
INSTALL_XSPEC="no"
INSTALL_XS_LITE="no"
INSTALL_ROOT="no"
BATCH="no"

while [ "${1:-}" != "" ]; do
    case "$1" in
      "--with-xspec")
        INSTALL_XSPEC="yes"
        ;;
      "--with-xspec-lite")
        INSTALL_XS_LITE="yes"
        ;;
      "--with-root")
        INSTALL_ROOT="yes"
        ;;
      "--batch")
        BATCH="yes"
        ;;
      "-h" | "--help")
        echo "install_3ML.sh [--with-xspec] [--with-root] [-h] [--help] [--batch]" && exit 0
        ;;
    esac
    shift
  done

# (xspec-lite is an hidden option on purpose, it is only meant to be used by Travis)

echo "Options:"
echo "--------"
echo "Installing xspec:                              "${INSTALL_XSPEC}
echo "Installing root :                              "${INSTALL_ROOT}
echo "Batch execution (assume yes to all questions): "${BATCH}
echo ""

# Make a small download script in Python to avoid dependencies on 
# utilities such as wget
rm __download.py >& /dev/null

cat > __download.py <<- EOM
import sys

try:
    
    # Python 2
    
    from urllib import urlretrieve

except AttributeError:
    
    # Python 3
    
    from urllib.request import urlretrieve

urlretrieve(sys.argv[1], sys.argv[2])
EOM

# Guess OS

os_guessed="unknown"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
        
        # Linux
        
        os_guessed="linux"


elif [[ "$OSTYPE" == darwin* ]]; then
        
        # Mac OSX
        
        os_guessed="osx"
        

elif [[ "$OSTYPE" == "cygwin" ]]; then
        
        # POSIX compatibility layer and Linux environment emulation for Windows
        
        os_guessed="linux"
        
else

        # Unknown.
        
        echo "Could not guess your OS. Exiting."
        
        exit 1
        
fi

echo "Running on ${os_guessed}"

# Function to install conda if needed
install_conda() {

    line
    echo "Installing conda"
    line

    # Make sure bunzip2 is installed (it's needed by the conda installer)
    if which bunzip2 >& /dev/null ; then
        
        echo "bunzip2 found"
    
    else
        
        echo "You need to install bzip2 first. Use your package manager."
        
        exit 3
    
    fi
    
    if [[ "$os_guessed" == "linux" ]]; then
            
            # Linux
            
            python __download.py https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh Miniconda2-latest.sh
    
    elif [[ "$os_guessed" == "osx" ]]; then
            
            # Mac OSX
            
            python __download.py https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh Miniconda2-latest.sh
            
            
    else
    
            # Unknown.
            
            echo "Should never get here. This is a bug."
            
            exit 100
            
    fi
    
    if bash Miniconda2-latest.sh -p ~/miniconda2 -b -u ; then
    
        echo "Installation of Conda successful"
    
    else
        
        echo "Could not install Conda. Please check errors above"
        
        exit 2
    
    fi
    
    rm -rf Miniconda2-latest.sh
    
}

# Function to generate the setup scripts
generate_init_script() {

# NOTE: the ${PATH} env variable when this function is called already contains
# the right path settings. It is substituted in the following expressions so
# the script will explicitly contains the paths

    cat > threeML_init.sh <<- EOM

export PATH=${PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
source activate threeML

EOM

    cat > threeML_init.csh <<- EOM

setenv OMP_NUM_THREADS 1
setenv MKL_NUM_THREADS 1
setenv NUMEXPR_NUM_THREADS 1
setenv CONDA_ENVS_PATH $(conda info | grep "envs directories" | cut -f2 -d":" )

source ${CONDA_PREFIX}/bin/deactivate.csh >& /dev/null
source ${CONDA_PREFIX}/bin/activate.csh threeML

EOM
 
}

line() {
   
   echo ""
   echo "###############################################"
   echo ""
}


line
echo "Installing/checking Conda installation"
line

# Guess whether we need to install Miniconda or not
if conda --version >& /dev/null ; then
    
    # Gather conda default environment path
    
    conda_path=$(conda info | grep "default environment" | cut -f2 -d":" | cut -f2 -d" ")
    
    echo "Found an already existing installation of conda in ${conda_path}"

else

    echo "I did not find a conda installation."

    if [[ "${BATCH}" == "no" ]]; then

        echo "If you do have an installation of conda and you want to use that, answer 'no' to the next question (the script will exit), make sure the 'conda' executable is in your PATH and then re-run the script."
        echo "If you do not have conda, then answer yes to the following question and I'll download it for you (it's free)"

        while true; do
            read -p "Do you wish to install Conda ? (yes/no) " yn
            case    $yn in
                [Yy]* ) break;;
                [Nn]* ) exit;;
                * ) echo "Please answer yes or no.";;
            esac
        done

    fi
    
    # If we are here, we need to install conda
    
    conda_path=${HOME}/miniconda2
    
    install_conda
    
fi

line
echo "Installing 3ML"
line

PACKAGES_TO_INSTALL="threeml"

if [[ "${INSTALL_XSPEC}" == "yes" ]]; then

    PACKAGES_TO_INSTALL="${PACKAGES_TO_INSTALL} xspec-modelsonly"

fi

if [[ "${INSTALL_XS_LITE}" == "yes" ]]; then

    PACKAGES_TO_INSTALL="${PACKAGES_TO_INSTALL} xspec-modelsonly-lite"

fi

if [[ "${INSTALL_ROOT}" == "yes" ]]; then

    PACKAGES_TO_INSTALL="${PACKAGES_TO_INSTALL} root5"

fi

# Now we have conda installed, let's install 3ML

export PATH=${conda_path}/bin:${PATH}

source deactivate

conda create --name threeML -y -c conda-forge -c threeml python=2.7 numpy scipy matplotlib ${PACKAGES_TO_INSTALL}

line
echo "Generating setup scripts"
line


generate_init_script ${conda_path}

if [ -n "${PYTHONPATH+set}" ]; then
  
  echo ""
  echo 'WARNING: it looks like you have a PYTHONPATH variable set. This could interfere with the working of Conda and 3ML.'
  echo ""

fi

if [ -n "${LD_LIBRARY_PATH+set}" ]; then

  echo ""
  echo '\nWARNING: it looks like you have a LD_LIBRARY_PATH variable set. This could interfere with the working of Conda and 3ML.\n'
  echo ""

fi

if [ -n "${DYLD_LIBRARY_PATH+set}" ]; then

  echo ""
  echo '\nWARNING: it looks like you have a DYLD_LIBRARY_PATH variable set. This could interfere with the working of Conda and 3ML.\n'
  echo ""

fi

# Cleanup
rm -rf __download.py


# Finally dump the .csh scripts to activate/deactivate in csh/tcsh and copy them to the miniconda installation
cat > activate.csh <<- "EOM"
#!/bin/csh

# Get the name of this script
set script_name = `basename $0`

# Get arguments
if ( $#argv < 1 ) then
	echo ""
	echo "Usage: source $script_name <CONDAENV>"
	exit 2
endif
set conda_env = $1

# Make sure the $CONDA_ENVS_PATH env var is defined
if ( ! $?CONDA_ENVS_PATH ) then
	echo ""
	echo 'You must set the environment variable $CONDA_ENVS_PATH to point to the parent directory containing your conda environments\n'
	echo "Usage: source $script_name <CONDAENV>"
	exit 2
endif

# Make sure the $CONDA_ENVS_PATH env var isn't empty
if ( "$CONDA_ENVS_PATH" == "" ) then
	echo ""
	echo "You must set the environment variable \$CONDA_ENVS_PATH to point to the parent directory containing your conda environments\n\n"
	echo "Usage: source $script_name <CONDAENV>"
	exit 2
endif

# See if the given Anaconda environment exists under $CONDA_ENVS_PATH
if ( ! -d "$CONDA_ENVS_PATH/$conda_env" ) then
	echo ""
	echo "The '$conda_env' conda environment was not found in $CONDA_ENVS_PATH"
	echo ""
	echo "Did you create one with 'conda create -n <myenv> python'?"
	exit 2
endif

# Remove duplicates from $PATH
set new_path = `echo $PATH | sed -e 's/$/:/;s/^/:/;s/:/::/g;:a;s#\(:[^:]\{1,\}:\)\(.*\)\1#\1\2#g;ta;s/::*/:/g;s/^://;s/:$//;'`

# Determine the active python environment
set active_python=`which python`

# If the active python environment is the production environment
set python_bin_dir=`which python | sed 's|/python$||'`
set test=`echo $active_python | awk -v test="$CONDA_ENVS_PATH" '$0 ~ test { print "MATCH" }'`
if ( $test != "MATCH" ) then
	setenv CONDA_PROD_ENV_BIN $python_bin_dir
	setenv PATH `echo $PATH | sed -e 's|^'$python_bin_dir':||' -e 's|:'$python_bin_dir':|:|' -e 's|:'$python_bin_dir'$||'`
	# Prepend the name of the conda environment to the prompt
	set prompt="($conda_env) $prompt"
# If the active python environment is a conda environment
else
	# See if this conda environment is already active
	set prev_conda_env=`which python | sed -e 's|^'$CONDA_ENVS_PATH'/||' -e 's|/bin/python$||'`
	if ( $prev_conda_env == $conda_env ) then
		echo ""
		echo "The '$conda_env' conda environment is already active"
		exit 0
	endif
	# Change the name of the conda environment in the prompt
	
	set prompt=`echo $prompt | sed 's|^('$prev_conda_env')|\('$conda_env'\)|'`
	# Remove the current conda environment from $PATH
	setenv PATH `echo $PATH | sed -e 's|^'$python_bin_dir':||' -e 's|:'$python_bin_dir':|:|' -e 's|:'$python_bin_dir'$||'`
endif

# Prepend $CONDA_ENVS_PATH/$conda_env/bin to the $PATH variable
setenv PATH $CONDA_ENVS_PATH/$conda_env/bin:$PATH

# set the CONDA_PREFIX path
setenv CONDA_PREFIX $CONDA_ENVS_PATH/$conda_env

# Print help info
echo "Your Python environment has been changed to the '$conda_env' conda environment. Here's the active version of Python:"
which python
python --version
echo "To switch back to your default Python environment, type 'source deactivate.csh'"
EOM

cat > deactivate.csh <<- "EOM"
#!/bin/csh

# Make sure the $CONDA_ENVS_PATH env var is defined
if ( ! $?CONDA_ENVS_PATH ) then
	echo ""
	echo 'You must set the environment variable $CONDA_ENVS_PATH to point to the parent directory containing your Anaconda environments\n'
	exit 2
endif

# Make sure the $CONDA_ENVS_PATH env var isn't empty
if ( "$CONDA_ENVS_PATH" == "" ) then
	echo ""
	echo "You must set the environment variable \$CONDA_ENVS_PATH to point to the parent directory containing your Anaconda environments\n\n"
	exit 2
endif

# Make sure the $CONDA_PROD_ENV_BIN env var is defined
if ( ! $?CONDA_PROD_ENV_BIN ) then
	echo ""
	echo 'Something went wrong with your Python environment. Try opening a new terminal to start with a fresh environment.\n'
	exit 2
endif

# Make sure the $CONDA_PROD_ENV_BIN env var isn't empty
if ( "$CONDA_PROD_ENV_BIN" == "" ) then
	echo ""
	echo "Something went wrong with your Python environment. Try opening a new terminal to start with a fresh environment.\n\n"
	exit 2
endif

# Get the current python binary
set python_path=`which python`

# See if the current python binary is found under $CONDA_ENVS_PATH, exit if not
set test=`echo $python_path | awk -v test="$CONDA_ENVS_PATH" '$0 ~ test { print "MATCH" }'`
if ( $test != "MATCH" ) then
	echo "You're not currently using a conda environment"
	exit 0
endif

# Remove all occurrences of this python binary path from the $PATH
if ( $test == "MATCH" ) then
	set python_bin_dir=`echo $python_path | sed 's|/python$||'`
	setenv PATH `echo $PATH | sed -e 's|^'$python_bin_dir':||' -e 's|:'$python_bin_dir':|:|' -e 's|:'$python_bin_dir'$||'`
endif

# Add the production conda environment to the $PATH
setenv PATH ${CONDA_PROD_ENV_BIN}:${PATH}

# Get the name of the current conda environment
set conda_env=`echo $python_path | sed -e 's|^'$CONDA_ENVS_PATH'/||' -e 's|/bin/python$||'`

# Remove the name of the conda environment from the prompt
set prompt=`echo $prompt | sed 's|^('$conda_env')||'`

# Print help info
echo "Your Python environment has been reset. Here's the active version of Python:"
which python
python --version
EOM

source activate threeML

mv activate.csh $CONDA_PREFIX/bin
mv deactivate.csh $CONDA_PREFIX/bin

source deactivate
