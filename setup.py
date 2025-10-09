import re
import os
from setuptools import setup, find_packages


# =============================================================================
# helper functions to extract meta-info from package
# =============================================================================
def read_version_file(*parts):
    return open(os.path.join(*parts), 'r').read()

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def find_version(*file_paths):
    version_file = read_version_file(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

def find_name(*file_paths):
    version_file = read_version_file(*file_paths)
    version_match = re.search(r"^__name__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find name string.")

def find_author(*file_paths):
    version_file = read_version_file(*file_paths)
    version_match = re.search(r"^__author__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find author string.")


# =============================================================================
# setup    
# =============================================================================
setup(
    name = find_name("CorrDis", "__init__.py"),
    version = find_version("CorrDis", "__init__.py"),
    author = find_author("CorrDis", "__init__.py"),
    author_email='denis.langevin@univ-amu.fr',
    license='MIT',
    packages=['CorrDis'],
    include_package_data=True,   # add data folder containing material dispersions
    description = ("A correlated disorder library"),
    long_description=read('README.md'),
    url='https://github.com/Milloupe/CorrDis',
    keywords=['Diffraction', 'Disorder', 'Correlation'],
    install_requires=[
          'numpy','matplotlib','scipy'
      ],

)
