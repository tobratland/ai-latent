#!/bin/bash

# Define the train folder
TRAIN_DIR="train"

# Create the train folder if it doesn't exist
mkdir -p "$TRAIN_DIR"

# Loop through all subfolders
for folder in */; do
    # Skip the train folder
    if [[ "$folder" == "$TRAIN_DIR/" ]]; then
        continue
    fi

    # Remove trailing slash from folder name
    folder_name="${folder%/}"

    # Move each file, renaming it with the folder name as prefix
    for file in "$folder"*; do
        if [[ -f "$file" ]]; then  # Ensure it's a file
            mv "$file" "$TRAIN_DIR/${folder_name}_$(basename "$file")"
        fi
    done
done

