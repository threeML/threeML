#!/usr/bin/env python

from setuptools import setup

import versioneer

setup(
    cmdclass=versioneer.get_cmdclass(),
    version=versioneer.get_version(),
)
