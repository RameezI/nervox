#!/bin/bash
 
 set -euo pipefail
set -x  # Enable debugging output to see each command being executed

# Function to output errors and exit
handle_error() {
    echo "Error occurred in step: $1"
    echo "Exiting due to previous errors."
    exit 1
}

# Capture Python files from the diff
files_formated=$(git diff --name-only | grep -E '.*\.py$' || true)
if [ $? -ne 0 ]; then handle_error "Capturing Python files"; fi

# Check if any Python files were captured
if [ -z "$files_formated" ]; then
    echo "No Python files changed."
    echo "files_formated=" >> $GITHUB_OUTPUT 
    echo "hunks=0" >> $GITHUB_OUTPUT
    exit 0  # Exit gracefully if no files changed
fi

hunks=$(git diff --unified=0 --name-only | grep -E '.*\.py$' | xargs -I {} git diff --unified=0 {} | grep '^@@' | wc -l)
if [ $? -ne 0 ]; then handle_error "Counting hunks"; fi

# get a string of the files nicely formatted so that we can use it in the next steps
# in the workflow using the ${{ steps.<step_id>.outputs.<output_id> }} syntax
# files_formatted=$(echo $files_formated | tr '\n' ' ')


echo "files_formatted="$files_formatted"">> $GITHUB_OUTPUT
echo "hunks=$hunks" >> $GITHUB_OUTPUT

