#!/bin/bash

# =================================== Variables ==============================================
# The following 3 variables correspond to the locations of the files downloaded from MIDRC
## 1) The directory containing the image files
DATA_DIR="/gpfs_projects/ravi.samala/DATA/MIDRC3/20221010_open_A1_CRDX_unzip"

## 2) The tsv file with case information
CASE_TSV="/gpfs_projects/alexis.burgon/bias.myti.report_testing_files/20221010_open_A1_all_Cases.tsv"

## 3) The tsv file with series information
SERIES_TSV="/gpfs_projects/alexis.burgon/bias.myti.report_testing_files/20221010_open_A1_all_Imaging_Series.tsv"

# The following 2 variables denote where the outputs should be saved
## 1) The summary json file that will be created
SUMMARY_JSON="data/open_A1_summary.json"

## 2) The directory in which the converted jpeg images will be saved
JPEG_DIR="data/open_A1_jpegs"

# The following variable denote the location where the tool repository is stored.
MAIN_DIR=$(dirname $(dirname $(realpath $0)))

# ============================================================================================

echo "Beginning data summarization (this may take a while)"
python ${MAIN_DIR}/src/utils/data_summarize.py \
    -i $DATA_DIR \
    -c $CASE_TSV \
    -s $SERIES_TSV \
    -o $SUMMARY_JSON
    
echo "Beginning image conversion..."
python ${MAIN_DIR}/src/utils/data_conversion.py \
    --save_dir $JPEG_DIR \
    --input_file $SUMMARY_JSON

echo "Data preprocessing complete!"