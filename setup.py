# Copyright(c) 2023 Rameez Ismail - All Rights Reserved
# Author: Rameez Ismail
# Email: rameez.ismaeel@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismail@philips.com
"""

from setuptools import setup

_MAJOR_VERSION = "0"
_MINOR_VERSION = "1"
_PATCH_VERSION = "0"

__version__ = ".".join(
    [
        _MAJOR_VERSION,
        _MINOR_VERSION,
        _PATCH_VERSION,
    ]
)

setup(
    name="nervox",
    version=__version__,
    description="""
      A framework to support development, training, evaluation & deployment
      of deep neural networks using tensorflow2.x.
      """,
    url="",
    author="Rameez Ismail",
    author_email="Rameez.ismaeel@gmail.com",
    packages=[],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "tensorflow==2.11.*",
        "tensorflow-datasets",
        "tensorflow-addons",
    ],
    zip_safe=False,
)
