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
Email: rameez.ismaeel@gmail.com
"""

"""
Universal launcher for tf-tests

# Copyright 2018 Royal-Philips. All Rights Reserved.
# Author: Rameez Ismail
"""

import argparse

# TODO: Add a universal launcher for all tests


def run_tests():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="verbosity level, use: [-v | -vv | -vvv]",
    )
    parser.add_argument(
        "-s", "--start-directory", default=None, help="directory to start discovery"
    )
    parser.add_argument(
        "-p",
        "--pattern",
        default="test*.py",
        help="pattern to match test files ('test*.py' default)",
    )
    parser.add_argument(
        "test", nargs="*", help="test specs (e.g. module.TestCase.test_func)"
    )
    args = parser.parse_args()
    raise NotImplemented


if __name__ == "__main__":
    # NOTE: True(success) -> 0, False(fail) -> 1
    run_tests()
