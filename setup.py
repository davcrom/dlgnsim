"""dlgnsim installation script

A "developer" install lets you work on or otherwise update ixtract in-place, in 
whatever folder you cloned it into with git, while still being able to call 
`import ixtract` and use it as a library system-wide. This creates an egg-link 
in your system site-packages or dist-packages folder to the source code:

>> sudo python3 setup.py develop

or the equivalent using pip:

$ sudo pip3 install -e .
"""

from setuptools import setup

setup(name='dlgnsim',
      license='BSD',
      description='simulations of dLGN with PyLGN',
      author='Davide Crombie'
      )
