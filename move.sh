#!/bin/bash

# Check if source and destination directories are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    exit 1
fi

source_dir="$1"
dest_dir="$2"

# Check if source directory exists
if [ ! -d "$source_dir" ]; then
    echo "Error: Source directory '$source_dir' does not exist"
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Move all contents and create symbolic links
for item in "$source_dir"/*; do
    if [ -e "$item" ]; then
        # Get the basename of the item
        base_name=$(basename "$item")
        
        # Move the item to destination
        mv "$item" "$dest_dir/"
        
        echo "Moved '$base_name' to '$dest_dir' and created symbolic link"
    fi
done
rm -r "$source_dir"
 # Create symbolic link
ln -s "$dest_dir" "$source_dir"

echo "Done! All contents moved and symbolic links created."