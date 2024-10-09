#!/bin/bash
if git diff --name-only | grep -qE '.*\.py$'; then
    echo "python_changes=true" >> $GITHUB_OUTPUT
    hunks_changed=$(git diff --unified=0 --name-only | grep -E '.*\.py$' | xargs -I {} git diff --unified=0 {} | grep '^@@' | wc -l)
    files_changed=$(git diff --name-only | grep -E '.*\.py$' | wc -l)
    echo "hunks=$hunks_changed" >> $GITHUB_OUTPUT
    echo "files=$files_changed" >> $GITHUB_OUTPUT
else
    echo "python_changes=false" >> $GITHUB_OUTPUT
fi