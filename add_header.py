# Copyright © 2023 Rameez Ismail - All Rights Reserved
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
#
# Author (s): Rameez Ismail
# Email (s):  rameez.ismaeel@gmail.com


import io
import argparse
from pathlib import Path

AUTHOR_NAME = "Rameez Ismail"
EMAIL = "rameez.ismaeel@gmail.com"

LICENSE = 'Apache License, Version 2.0 (the "License")'
LICENSE_URL = " http://www.apache.org/licenses/LICENSE-2.0"

ATTRIBUTED_LICENSE = "Apache License, Version 2.0"
ATTRIBUTED_LICENSE_URL = " http://www.apache.org/licenses/LICENSE-2.0"

PROJECT_NAME = "project_name"
PROJECT_URL = ""


def get_license_header(
    author: str = AUTHOR_NAME,
    email: str = EMAIL,
    license: str = LICENSE,
    license_url: str = LICENSE_URL,
):
    return f"""
# Copyright © 2023 Rameez Ismail - All Rights Reserved
# Licensed under the {license};
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#   {license_url}
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author(s): {author}
# Email(s):  {email}
"""


def get_attribution_header(
    project_name: str = PROJECT_NAME,
    project_url: str = PROJECT_URL,
    license: str = ATTRIBUTED_LICENSE,
    license_url: str = ATTRIBUTED_LICENSE_URL,
):
    return f"""
# \n# This code is adapted from {project_name}: 
# {project_url}
# The project is licensed under the {license}. 
# You may obtain a copy of the license at:
# {license_url}
"""


def add_header(file_path):
    # Read the contents of the file
    with io.open(file_path, "r", encoding="utf8") as f:
        content = f.read()

    # Check if the file already has a license header
    if content.strip().startswith("#"):
        skip_lines = True
        _content = ""
        for line in content.splitlines():
            if line.strip().startswith("#") and skip_lines:
                continue
            else:
                skip_lines = False
                _content += line + "\n"
        content = _content.strip()

    # Prepend the license header to the file contents
    if args.add_attribution:
        header = (
            get_license_header(
                args.author, args.email, args.license, args.license_url
            ).strip()
            + "\n"
            + get_attribution_header(
                args.attributed_project_name,
                args.attributed_project_url,
                args.attributed_license,
                args.attributed_license_url,
            ).strip()
            + "\n\n"
        )
    else:
        header = (
            get_license_header(
                args.author, args.email, args.license, args.license_url
            ).strip()
            + "\n\n"
        )

    content = header + content

    # Write the modified contents back to the file
    with io.open(file_path, "w", encoding="utf8") as f:
        f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="the root directory which is searched recursively for python files",
    )
    group.add_argument("--file", type=str, default=None)
    parser.add_argument(
        "--add_attribution", required=False, default=False, action="store_true"
    )
    parser.add_argument("--author", required=False, type=str, default=AUTHOR_NAME)
    parser.add_argument("--email", required=False, type=str, default=EMAIL)
    parser.add_argument("--license", required=False, type=str, default=LICENSE)
    parser.add_argument("--license_url", required=False, type=str, default=LICENSE_URL)
    parser.add_argument(
        "--attributed_license", required=False, type=str, default=ATTRIBUTED_LICENSE
    )
    parser.add_argument(
        "--attributed_license_url",
        required=False,
        type=str,
        default=ATTRIBUTED_LICENSE_URL,
    )

    parser.add_argument(
        "--attributed_project_name", required=False, type=str, default=PROJECT_NAME
    )

    parser.add_argument(
        "--attributed_project_url", required=False, type=str, default=PROJECT_URL
    )

    args = parser.parse_args()

    if args.root_dir is None and args.file is None:
        raise ValueError("Either --root_dir or --file must be specified")

    if args.file is not None:
        python_files = [Path(args.file)]

    else:
        root_path = Path(args.root_dir)
        python_files = list(
            root_path.glob(
                "**/*.py",
            )
        )

    for file in python_files:
        add_header(file)
        print(f"processed: {file}")
