"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismail@philips.com
"""

from setuptools import setup

_MAJOR_VERSION = '0'
_MINOR_VERSION = '1'
_PATCH_VERSION = '0'

__version__ = '.'.join([
    _MAJOR_VERSION,
    _MINOR_VERSION,
    _PATCH_VERSION,
])

setup(name='nervox',
      version=__version__,
      description="""
      A framework to support development, training, evaluation & deployment
      of deep neural networks using tensorflow2.x.
      """,
      url='',
      author='Rameez Ismail',
      author_email='Rameez.ismail@philips.com',
      license='@Copyrights [Royal Philips], all rights reserved',
      packages=[],
      install_requires=['numpy', 'pandas', 'matplotlib', 'scikit-learn',
                        'tensorflow==2.11.0', 'tensorflow-datasets',
                        'tensorflow-addons'],
      zip_safe=False)
