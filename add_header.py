"""
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
"""

import os
import argparse
from pathlib import Path

name = 'Rameez'
email = 'rameez.ismaeel@gmai.com'

HEADER = """\"\"\"
Copyright (C) 2021 Rameez Ismail - All Rights Reserved
Author: Rameez Ismail
Email: rameez.ismaeel@gmail.com
\"\"\"\n\n"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=False, type=str, default='.',
                        help='the root directory which is searched recursively for python files')
    args = parser.parse_args()
    root_path = Path(args.root_dir)
    python_files = list(root_path.glob('**/*.py', ))
    
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
