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
