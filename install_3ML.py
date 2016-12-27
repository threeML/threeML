import subprocess
import sys
from distutils.util import strtobool
import re
import os
import socket


def internet_connection_is_active(host="8.8.8.8", port=53, timeout=3):
    """
    Check that a internet connection is working by trying contacting the following host:

    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """

    try:

        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))

    except Exception as ex:

        print(ex.message)
        return False

    else:

        return True


def discovery_message(message):

    print("\n * %s\n" % message)


def fixable_problem(message):

    print("\nPROBLEM: %s\n" % message)


def fatal_error(message):

    print("\n\nFATAL: %s\n\n" % message)

    sys.exit(-1)


def yes_or_no(prompt):

    while True:

        # This is only python2 compatible. Will need to change to input() instead of raw_input() with python3.

        answer = raw_input(prompt)

        # strtobool returns True if answer is yes,y,t,1, no if

        try:

            return_bool = strtobool(answer)

        except ValueError:

            print("Invalid answer. Please use one of yes,y,t,1 to say yes or no,n,f,0 to say no. Try again.")

        else:

            return return_bool


def prompt_string(message, default=None, path=False):

    while True:

        answer = raw_input(message)

        if answer == '':

            answer = default

        if answer is None: # this gets triggered if there was no default

            print("You have to provide an answer")
            continue

        else:

            # validate answer
            if path:

                if len(re.findall("[^0-9a-zA-Z_%s~${}]" % os.path.sep, answer)) > 0:

                    print("Invalid path. Please use only letters, numbers and underscores (and %s)" % os.path.sep)
                    continue

                else:
                    # Valid answer
                    return answer

            else:

                if len(re.findall("[^0-9a-zA-Z_]", answer)) > 0:

                    print("Invalid answer. Please use only letters, numbers and underscores")
                    continue

                else:
                    # Valid answer
                    return answer


if __name__ == "__main__":

    # Make sure we are running in python2, not python3

    try:

        assert hasattr(__builtins__, "raw_input")

    except AssertionError:

        message = "You tried running the install script with python3. Please use python2 instead. Usually this can " \
                  "be achieved with 'python2 install_3ML.py' " \
                  "(instead of just 'python install_3ML.py' or './install_3ML.py')"

        fatal_error(message)

    # Make sure we are connected to the internet
    if not internet_connection_is_active():

        fatal_error("Looks like you are not connected to the internet right now. Check your connection and try again.")

    # Ask initial confirmation

    print("\nThis script installs 3ML and astromodels (as well as cthreeML if your system supports it, and all "
          "dependencies) in a new python virtual environment. This way it will not interfere with your system python "
          "or other python versions you might have installed.")
    print("(see http://docs.python-guide.org/en/latest/dev/virtualenvs/ for info on virtual environments)")
    print("\nPLEASE NOTE: you need to have set up the environments for all experiments you want to use in 3ML *before* "
          "running this script. If you didn't do so please fix this before continuing.")

    choice = yes_or_no("\nContinue (yes/no)? ")

    if not choice:

        sys.exit(-2)


    # First make sure pip is installed
    try:

        pip_version_string = subprocess.check_output("pip --version", shell=True)

    except subprocess.CalledProcessError:

        message = "Could not execute 'pip --version'. Likely you do not have pip installed. " \
                  "Please install pip first, then re-run this script. Use your package manager (preferred), or " \
                  "follow instructions at https://pip.pypa.io/en/stable/installing/"

        fatal_error(message)

    else:

        discovery_message("Found pip with this version string: %s" % pip_version_string)

    # Then make sure git is installed
    try:

        git_version_string = subprocess.check_output("git --version", shell=True)

    except subprocess.CalledProcessError:

        message = "Could not execute 'git --version'. Likely you do not have git installed. " \
                  "Use your package manager to install git first, then re-run this script."

        fatal_error(message)

    else:

        discovery_message("Found git with this version string: %s" % git_version_string)

    # Check that virtualenv is installed, otherwise try to install it
    try:

        venv_version_string = subprocess.check_output("virtualenv --version", shell=True)

        # If the previous line worked, virtualenv does not need any specific path to work

        virtual_env_bin = 'virtualenv'

    except subprocess.CalledProcessError:

        message = "virtualenv is not installed. Please install it before running this script. You can do that by " \
                  "running 'pip install virtualenv'. If you do not have admin rights you can still install it as " \
                  "'pip install --user virtualenv'. However, remember to add the path where the virtualenv executable " \
                  "gets installed (usually ~/.local/bin/) to your PATH evironment variable before running this " \
                  "script again."

        fatal_error(message)

    # Here we have a working pip and virtualenv

    # Create the virtual environment
    default_path = os.path.abspath(os.path.expanduser(os.path.join("~", "3ML_env")))

    env_path = prompt_string("Please provide a path for your virtual environment "
                             "(hit enter to use default: %s): " % default_path,
                             default=default_path, path=True)
    env_path = os.path.abspath(os.path.expandvars(os.path.expanduser(env_path)))

    discovery_message("Creating virtual environment at %s" % env_path)

    # Assert it does not exist
    if os.path.exists(env_path):

        fatal_error("Path %s exists. Are you sure you did not run this script already? If you want to activate the "
                    "environment, use 'source %s'. If you want to start from scratch, "
                    "remove %s first." % (env_path, os.path.join(env_path,'bin','activate'), env_path))

    subprocess.check_call("virtualenv --no-site-packages %s" % env_path, shell=True)

    # Write a script which will run in the new environment
    temp_file = "__install_script.sh"

    with open(temp_file, "w+") as f:

        f.write("#/bin/bash\n")
        f.write("source %s/bin/activate\n" % env_path)
        f.write("pip install numpy scipy matplotlib iminuit astropy ipython ipyparallel --upgrade\n")
        f.write("pip install git+https://github.com/giacomov/3ML.git --upgrade\n")
        f.write("pip install git+https://github.com/giacomov/astromodels.git --upgrade\n")
        f.write("pip install git+https://github.com/giacomov/cthreeML.git || "
                "echo '\n\nNOTE: could not install chtreeML. Probably boost python is not available' \n")

    # Execute script
    subprocess.check_call("/bin/bash __install_script.sh", shell=True)

    # Remove script
    os.remove("__install_script.sh")

    # Print final message
    discovery_message("Installation complete.")

    print("\n\nREMEMBER: before using 3ML you need to run 'source %s/bin/activate'. Normally this needs to "
          "be done after you set up all the other packages, like AERIE, the Fermi Science Tools and so on, otherwise"
          "some of the plugins might be unavailable.\n "
          "If you want to uninstall, simply remove the entire directory %s "
          "(no other python environment will be touched, but of course you will loose anything you have "
          "installed in that environment)" % (env_path, env_path))

    sys.exit(0)


