#!/bin/bash

# Colors
RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default settings
DATA_DIR=$1
DATASET_DIR=$DATA_DIR/multimodal_har_datasets/
UTD_MHAD_DIR=utd_mhad
DOWNLOAD_UTD_MHAD=true

function download_utd_mhad () {
  echo
  echo -e "${CYAN}Downloading UTD-MHAD:${NC}"
  echo -e "Inertial Data 1/2..."
  wget http://www.utdallas.edu/~kehtar/UTD-MAD/Inertial.zip -P "$DATASET_DIR"
  echo -e "Skeleton Data 2/2..."
  wget http://www.utdallas.edu/~kehtar/UTD-MAD/Skeleton.zip -P "$DATASET_DIR"
}

function extract_utd_mhad () {
  echo
  echo -e "${CYAN}Extracting UTD-MHAD:${NC}"
  mkdir -p "$DATASET_DIR/$UTD_MHAD_DIR"
  (cd "$DATASET_DIR" && unzip Inertial.zip -d $UTD_MHAD_DIR/)
  (cd "$DATASET_DIR" && unzip Skeleton.zip -d $UTD_MHAD_DIR/)
}

function clean_zip_files () {
  (cd "$DATASET_DIR" && rm -f *.zip)
}


# Execute main program
if [ "$DOWNLOAD_UTD_MHAD" == "true" ]; then
  if [ -d "$DATASET_DIR/$UTD_MHAD_DIR" ]; then
    echo -e "${YELLOW}[Warning]${NC}"
    echo -en "\tIt seems like utd-mhad is already downloaded under the directory: "
    echo -e "${CYAN}$DATASET_DIR/$UTD_MHAD_DIR${NC}"
    echo -e "\tRemove it if you want to download the dataset either way"
  else
    download_utd_mhad
    extract_utd_mhad
  fi
fi

clean_zip_files

