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

import os
import io
import argparse
from pathlib import Path

NAME = 'Rameez Ismail'
EMAIL = 'rameez.ismaeel@gmail.com'

LICENSE = 'Apache License, Version 2.0 (the "License")'
LICENSE_LINK = ' http://www.apache.org/licenses/LICENSE-2.0'

ATTRIBUTED_LICENSE = 'Apache License, Version 2.0 (the "License")'
ATTRIBUTED_LICENSE_LINK = ' http://www.apache.org/licenses/LICENSE-2.0'


# Define the license header as a string
LICENSE_HEADER = f"""
# Copyright(c) 2023 {NAME} - All Rights Reserved
# Author: {NAME}
# Email: {EMAIL}
#
# Licensed under the {LICENSE};
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   {LICENSE_LINK}
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

ATTRIBUTION_HEADER = f"""
# This code is adapted from the [project name] project, licensed under the
# {ATTRIBUTED_LICENSE}. You may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#   {ATTRIBUTED_LICENSE_LINK}
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License."""

def add_license_header(file_path, renew:bool=False):

     # Read the contents of the file
    with io.open(file_path, 'r', encoding='utf8') as f:
        content = f.read()

    if renew:
        renewed_content = ''
        skip_lines = False
        for line in content.splitlines():
            skip_lines = True if 'Copyright(c)' \
                and line.strip().startswith('#') else False
            
            if line.strip().startswith('#') and skip_lines:
                continue

            else:
                skip_lines = False
                renewed_content += line + '\n'
    else:
        renewed_content = content

    # Check if the file already starts with the license header
    if renewed_content.startswith(LICENSE_HEADER.strip()):
        return

    # Prepend the license header to the file contents
    renewed_content = LICENSE_HEADER.strip() + '\n\n' + renewed_content

    # Write the modified contents back to the file
    with io.open(file_path, 'w', encoding='utf8') as f:
        f.write(renewed_content)

    # Read the contents of the file
    with io.open(file_path, 'r', encoding='utf8') as f:
        content = f.read()

    # Check if the file already starts with the license header
    if content.startswith(LICENSE_HEADER.strip()):
        return

    # Prepend the license header to the file contents
    renewed_content = LICENSE_HEADER.strip() + '\n\n' + content

    # Write the modified contents back to the file
    with io.open(file_path, 'w', encoding='utf8') as f:
        f.write(renewed_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--root_dir', required=False, type=str, default=None,
                        help='the root directory which is searched recursively for python files')
    group.add_argument('--file', required=False, type=str, default=None)
    parser.add_argument('--add_attribution', required=False, type=bool,
                         default=False, action='store_true',)
   
    parser.add_argument('--name', required=False, type=str, default=NAME)
    parser.add_argument('--email', required=False, type=str, default=NAME)
    parser.add_argument('--license', required=False, type=str, default=LICENSE)
    parser.add_argument('--license_link', required=False, type=str, default=LICENSE_LINK)
    parser.add_argument('--attributed_license', required=False, type=str, default=ATTRIBUTED_LICENSE)
    parser.add_argument('--attributed_license_link', required=False, type=str, default=ATTRIBUTED_LICENSE_LINK)
    args = parser.parse_args()

    if args.root_dir is None and args.file is None:
        raise ValueError('Either --root_dir or --file must be specified')

    if args.file is not None:
        python_files = [Path(args.file)]

    else:    
        root_path = Path(args.root_dir)
        python_files = list(root_path.glob('**/*.py', ))

    # set license parameters
    NAME = args.name
    EMAIL = args.email
    LICENSE= args.license
    LICENSE_LINK = args.license_link
    ATTRIBUTED_LICENSE = args.attributed_license
    ATTRIBUTED_LICENSE_LINK = args.attributed_license_link

     
    for file in python_files:
        dummy_file = f'{str(file)}.bak'

        with open(str(file), 'r') as read_obj:
            lines = (line.rstrip() for line in read_obj)  # All lines including the blank ones
            lines = (line for line in lines if line)  # Non-blank lines
            is_header_present = next(lines).startswith('\"\"\"')
        
        with open(str(file), 'r') as read_obj, open(str(dummy_file), 'w') as write_obj:
            
            if not is_header_present:
                for line in iter(HEADER.splitlines()):
                    write_obj.write(line + '\n')
                    
            for line in read_obj:
                write_obj.write(line)
        
        os.remove(str(file))
        os.rename(str(dummy_file), str(file))
        print(f'processed: {file}')
