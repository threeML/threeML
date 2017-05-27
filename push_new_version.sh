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
  git commit -m 'Automatic patch number increase [ci skip]' 3ML/threeML/version.py
  git push
}

setup_git
checkout_repo
increment_version
commit_and_push
