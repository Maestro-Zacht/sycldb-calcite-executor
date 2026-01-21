#!/bin/bash

ssb_dir=$1

if [ -z "$ssb_dir" ]; then
    echo "Usage: $0 <ssb_directory>"
    exit 1
fi

if [ ! -d "$ssb_dir" ]; then
    echo "Error: Directory '$ssb_dir' does not exist."
    exit 1
fi

cd "$ssb_dir" || exit 1

# Remove unused columns (I'm leaving LINEORDER0 for the python code)
rm LINEORDER1 LINEORDER6 LINEORDER7 LINEORDER10 LINEORDER14 LINEORDER15 LINEORDER16