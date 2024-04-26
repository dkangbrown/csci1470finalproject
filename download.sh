#!/bin/bash

# Define a base directory
BASE_DIR="./data"

# Create the base directory if it doesn't exist
mkdir -p "$BASE_DIR"

# Function to download from specific URLs
download_data () {
    local url=$1
    local folder=$2
    local file=$3
    local folder_path="$BASE_DIR/$folder"  # Correct path variable for folder
    local path="$folder_path/$file"  # Ensure path includes folder_path

    # Create directory for folder if not exists
    mkdir -p "$folder_path"

    if [[ $url == *.git ]]; then
        # Check if the repository folder is already cloned and is not empty
        if [ -d "$folder_path" ] && [ "$(ls -A $folder_path)" ]; then
            echo "Git repository already cloned in $folder_path."
        else
            echo "Cloning Git repository..."
            git clone "$url" "$folder_path"
        fi
    else
        # For non-Git URL, check if the file already exists
        if [ -e "$path" ]; then
            echo "$file already exists at $path."
        else
            echo "Downloading $file..."
            curl -o "$path" -L "$url"
            # Check if the file is a ZIP file and handle accordingly
            if [[ $file == *.zip ]]; then
                echo "Unzipping $file..."
                unzip -o "$path" -d "$folder_path"  # Ensure unzipping into the correct directory
                echo "Removing $file..."
                rm "$path"
            fi
        fi
    fi
}

download_data "https://guitaralliance.com/chord-lyric-text.zip" "chord-lyric-text" "chord-lyric-text.zip"
download_data "git@github.com:00sapo/OpenEWLD.git" "OpenEWLD" "OpenEWLD"
