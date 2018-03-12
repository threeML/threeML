#!/bin/sh

setup_git() {
  git config --global user.email "travis@travis-ci.org"
  git config --global user.name "Travis CI"
}

checkout_repo() {
  git clone https://${GH_TOKEN}@github.com/giacomov/3ML.git
}

increment_version() {
  python 3ML/ci/set_minor_version.py --patch ${TRAVIS_BUILD_NUMBER} --version_file 3ML/threeML/version.py
}

commit_and_push() {
  cd 3ML
  git commit -m 'Automatic patch number increase [ci skip]' threeML/version.py
  git push
}

# Increment versions only for jobs which aren't pull requests

if [ "${TRAVIS_PULL_REQUEST}" = "false" ]; then

    setup_git
    checkout_repo
    increment_version
    commit_and_push

else
    
    echo "This is a pull request. Not incrementing version."

fi
